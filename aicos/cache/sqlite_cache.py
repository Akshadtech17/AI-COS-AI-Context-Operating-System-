"""
SQLite persistence layer for the semantic cache.

Schema holds prompts, responses, embeddings, and metadata.
Designed for sub-20ms exact lookups and batch similarity scans.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import String, Float, Integer, Text, DateTime, Index, select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from aicos.core.database import build_engine


class Base(DeclarativeBase):
    pass


class CacheRow(Base):
    __tablename__ = "cache_entries"
    __table_args__ = (
        Index("idx_cache_prompt_hash", "prompt_hash"),
        Index("idx_cache_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prompt_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    context_hash: Mapped[str] = mapped_column(String(64), default="")
    response: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String(100), default="")
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_hit_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    ttl_seconds: Mapped[int] = mapped_column(Integer, default=86400)

    @property
    def embedding(self) -> np.ndarray:
        return np.array(json.loads(self.embedding_json), dtype=np.float32)

    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds <= 0:
            return False
        # created_at may be timezone-naive (SQLite strips tz info)
        created = self.created_at
        if created.tzinfo is None:
            now = datetime.utcnow()
        else:
            now = datetime.now(timezone.utc)
        age = (now - created).total_seconds()
        return age > self.ttl_seconds


@dataclass
class CacheEntry:
    id: int
    prompt: str
    response: str
    embedding: np.ndarray
    model: str
    input_tokens: int
    output_tokens: int
    hit_count: int
    created_at: datetime


class SQLiteCache:
    def __init__(
        self,
        database_url: str,
        max_size: int = 10_000,
        ttl_seconds: int = 86400,
    ) -> None:
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._engine = build_engine(database_url)
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def initialize(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    @staticmethod
    def make_key(prompt: str, context_hash: str = "") -> str:
        content = f"{prompt}::{context_hash}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def get_exact(self, prompt_hash: str) -> CacheEntry | None:
        """Exact hash lookup — O(1) by index, target < 5ms."""
        async with self._session_factory() as session:
            stmt = select(CacheRow).where(CacheRow.prompt_hash == prompt_hash)
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()

            if row is None or row.is_expired:
                return None

            # Update hit count
            row.hit_count += 1
            row.last_hit_at = datetime.now(timezone.utc)
            await session.commit()

            return CacheEntry(
                id=row.id,
                prompt=row.prompt,
                response=row.response,
                embedding=row.embedding,
                model=row.model,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                hit_count=row.hit_count,
                created_at=row.created_at,
            )

    async def get_recent(self, limit: int = 1000) -> list[CacheEntry]:
        """
        Load recent non-expired entries for semantic similarity scan.
        Ordered by hit_count DESC, created_at DESC to prioritize popular entries.
        """
        async with self._session_factory() as session:
            stmt = (
                select(CacheRow)
                .order_by(CacheRow.hit_count.desc(), CacheRow.created_at.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        now = datetime.now(timezone.utc)
        return [
            CacheEntry(
                id=row.id,
                prompt=row.prompt,
                response=row.response,
                embedding=row.embedding,
                model=row.model,
                input_tokens=row.input_tokens,
                output_tokens=row.output_tokens,
                hit_count=row.hit_count,
                created_at=row.created_at,
            )
            for row in rows
            if not row.is_expired
        ]

    async def set(
        self,
        prompt: str,
        response: str,
        embedding: np.ndarray,
        context_hash: str = "",
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Store a cache entry, replacing if exists."""
        prompt_hash = self.make_key(prompt, context_hash)
        now = datetime.now(timezone.utc)

        async with self._session_factory() as session:
            existing = await session.scalar(
                select(CacheRow).where(CacheRow.prompt_hash == prompt_hash)
            )
            if existing:
                existing.response = response
                existing.embedding_json = json.dumps(embedding.tolist())
                existing.last_hit_at = now
            else:
                row = CacheRow(
                    prompt_hash=prompt_hash,
                    prompt=prompt,
                    context_hash=context_hash,
                    response=response,
                    embedding_json=json.dumps(embedding.tolist()),
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    hit_count=0,
                    created_at=now,
                    last_hit_at=now,
                    ttl_seconds=self._ttl_seconds,
                )
                session.add(row)

            await session.commit()

            # Prune if over max size
            count = await session.scalar(select(func.count()).select_from(CacheRow))
            if count and count > self._max_size:
                await self._prune(session, count - self._max_size)
                await session.commit()

    async def invalidate(self, prompt_hash: str) -> bool:
        async with self._session_factory() as session:
            row = await session.scalar(
                select(CacheRow).where(CacheRow.prompt_hash == prompt_hash)
            )
            if not row:
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def clear(self) -> int:
        async with self._session_factory() as session:
            count = await session.scalar(select(func.count()).select_from(CacheRow))
            await session.execute(delete(CacheRow))
            await session.commit()
            return int(count or 0)

    async def stats(self) -> dict[str, Any]:
        async with self._session_factory() as session:
            total = await session.scalar(select(func.count()).select_from(CacheRow))
            total_hits = await session.scalar(select(func.sum(CacheRow.hit_count)))
            return {
                "total_entries": int(total or 0),
                "total_hits": int(total_hits or 0),
                "max_size": self._max_size,
            }

    async def _prune(self, session: AsyncSession, count: int) -> None:
        stmt = (
            select(CacheRow)
            .order_by(CacheRow.hit_count.asc(), CacheRow.last_hit_at.asc())
            .limit(count)
        )
        result = await session.execute(stmt)
        for row in result.scalars():
            await session.delete(row)

    async def close(self) -> None:
        await self._engine.dispose()
