"""
Real-time cost tracker with per-session tracking and SQLite persistence.

Cost records are written to the database asynchronously (fire-and-forget)
so persistence never adds latency to request handling. On startup, the last
1 000 records are loaded to restore session-level counters.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, Float, Integer, String, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from aicos.analytics.metrics import get_metrics
from aicos.core.router import MODEL_REGISTRY


class _CostBase(DeclarativeBase):
    pass


class _CostRow(_CostBase):
    __tablename__ = "cost_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model: Mapped[str] = mapped_column(String(200))
    provider: Mapped[str] = mapped_column(String(100))
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    cost_usd: Mapped[float] = mapped_column(Float)
    task_type: Mapped[str] = mapped_column(String(100))
    session_id: Mapped[str | None] = mapped_column(String(200), nullable=True)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    recorded_at: Mapped[float] = mapped_column(Float)


@dataclass
class CostRecord:
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    task_type: str
    timestamp: datetime
    session_id: str | None = None
    cache_hit: bool = False


class CostTracker:
    """
    Tracks per-request cost data.

    Pass db_path to enable SQLite persistence; omit (or pass None) for
    in-memory-only mode (tests, CLI).
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._session_cost: float = 0.0
        self._session_tokens_in: int = 0
        self._session_tokens_out: int = 0
        self._session_requests: int = 0
        self._session_start: float = time.time()
        self._records: list[CostRecord] = []
        self._db_path = db_path
        self._engine: Any = None
        self._session_factory: Any = None

    async def initialize(self) -> None:
        """Create the cost_records table and load historical data. Call once at startup."""
        if not self._db_path:
            return
        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{self._db_path}",
            echo=False,
            pool_pre_ping=True,
        )
        async with self._engine.begin() as conn:
            await conn.run_sync(_CostBase.metadata.create_all)

        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

        # Load recent records to restore historical stats
        async with self._session_factory() as session:
            result = await session.execute(
                select(_CostRow)
                .order_by(_CostRow.recorded_at.desc())
                .limit(1000)
            )
            for row in result.scalars().all():
                self._records.append(CostRecord(
                    model=row.model,
                    provider=row.provider,
                    input_tokens=row.input_tokens,
                    output_tokens=row.output_tokens,
                    cost_usd=row.cost_usd,
                    task_type=row.task_type,
                    timestamp=datetime.fromtimestamp(row.recorded_at, tz=timezone.utc),
                    session_id=row.session_id,
                    cache_hit=bool(row.cache_hit),
                ))

    async def _persist(self, record: CostRecord) -> None:
        if not self._session_factory:
            return
        try:
            async with self._session_factory() as session:
                row = _CostRow(
                    model=record.model,
                    provider=record.provider,
                    input_tokens=record.input_tokens,
                    output_tokens=record.output_tokens,
                    cost_usd=record.cost_usd,
                    task_type=record.task_type,
                    session_id=record.session_id,
                    cache_hit=record.cache_hit,
                    recorded_at=record.timestamp.timestamp(),
                )
                session.add(row)
                await session.commit()
        except Exception:
            pass  # Never let persistence failure surface to the caller

    async def close(self) -> None:
        if self._engine:
            await self._engine.dispose()

    def compute_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        spec = MODEL_REGISTRY.get(model)
        if not spec:
            return (input_tokens * 1.0 + output_tokens * 3.0) / 1_000_000
        return (
            input_tokens * spec.input_cost_per_1m / 1_000_000
            + output_tokens * spec.output_cost_per_1m / 1_000_000
        )

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        task_type: str,
        session_id: str | None = None,
        cache_hit: bool = False,
    ) -> CostRecord:
        cost = 0.0 if cache_hit else self.compute_cost(model, input_tokens, output_tokens)
        spec = MODEL_REGISTRY.get(model)
        provider = spec.provider if spec else "unknown"

        record = CostRecord(
            model=model,
            provider=provider,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            task_type=task_type,
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            cache_hit=cache_hit,
        )

        self._records.append(record)
        self._session_cost += cost
        self._session_tokens_in += input_tokens
        self._session_tokens_out += output_tokens
        self._session_requests += 1

        get_metrics().record_request(
            model=model,
            task_type=task_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            latency_ms=0,
            cache_hit=cache_hit,
        )

        # Fire-and-forget async persistence (no-op when no event loop)
        try:
            asyncio.get_running_loop().create_task(self._persist(record))
        except RuntimeError:
            pass

        return record

    def compute_savings(self, cached_model: str, input_tokens: int, output_tokens: int) -> float:
        return self.compute_cost(cached_model, input_tokens, output_tokens)

    @property
    def session_cost(self) -> float:
        return self._session_cost

    @property
    def session_tokens(self) -> tuple[int, int]:
        return self._session_tokens_in, self._session_tokens_out

    @property
    def session_duration_seconds(self) -> float:
        return time.time() - self._session_start

    def session_summary(self) -> dict[str, object]:
        duration = self.session_duration_seconds
        return {
            "requests": self._session_requests,
            "input_tokens": self._session_tokens_in,
            "output_tokens": self._session_tokens_out,
            "total_tokens": self._session_tokens_in + self._session_tokens_out,
            "cost_usd": round(self._session_cost, 6),
            "duration_seconds": round(duration, 2),
            "cost_per_minute": (
                round(self._session_cost / (duration / 60), 6) if duration > 0 else 0
            ),
        }

    def breakdown_by_model(self) -> dict[str, dict[str, object]]:
        result: dict[str, dict[str, object]] = {}
        for r in self._records:
            if r.model not in result:
                result[r.model] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            result[r.model]["requests"] = int(result[r.model]["requests"]) + 1  # type: ignore[arg-type]
            result[r.model]["input_tokens"] = int(result[r.model]["input_tokens"]) + r.input_tokens  # type: ignore[arg-type]
            result[r.model]["output_tokens"] = int(result[r.model]["output_tokens"]) + r.output_tokens  # type: ignore[arg-type]
            result[r.model]["cost_usd"] = round(float(result[r.model]["cost_usd"]) + r.cost_usd, 6)  # type: ignore[arg-type]
        return result

    def cheapest_model_for_task(self, task_type: str) -> str | None:
        task_records = [r for r in self._records if r.task_type == task_type]
        if not task_records:
            return None
        return min(
            task_records,
            key=lambda r: r.cost_usd / max(r.input_tokens + r.output_tokens, 1),
        ).model
