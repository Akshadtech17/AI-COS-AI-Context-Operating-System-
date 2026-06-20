"""
Persistent memory store backed by SQLite via SQLAlchemy async.

Implements long-term memory with:
- Relevance scoring (recency + access frequency + cosine similarity)
- Tag-based filtering
- Automatic pruning when limit is exceeded
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import String, Float, Integer, Text, DateTime, select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from aicos.memory.embeddings import EmbeddingEngine


class Base(DeclarativeBase):
    pass


class MemoryItem(Base):
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_json: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    tags: Mapped[str] = mapped_column(String(1000), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    accessed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    access_count: Mapped[int] = mapped_column(Integer, default=0)
    relevance_score: Mapped[float] = mapped_column(Float, default=1.0)

    @property
    def embedding(self) -> np.ndarray:
        return np.array(json.loads(self.embedding_json), dtype=np.float32)

    @property
    def attrs(self) -> dict[str, Any]:
        return json.loads(self.metadata_json)

    @property
    def tag_list(self) -> list[str]:
        return [t.strip() for t in self.tags.split(",") if t.strip()]


class MemoryStore:
    """
    Async SQLite-backed memory store.

    Scoring formula:
        score = cosine_sim * 0.6 + recency_decay * 0.25 + access_freq * 0.15

    Recency decay: exp(-age_days / 30) — memories decay over 30 days
    Access frequency: log(1 + access_count) / log(1 + max_access)
    """

    RECENCY_HALF_LIFE_DAYS = 30.0
    COSINE_WEIGHT = 0.60
    RECENCY_WEIGHT = 0.25
    ACCESS_WEIGHT = 0.15

    def __init__(
        self,
        db_path: str | Path,
        embedding_engine: EmbeddingEngine | None = None,
        max_items: int = 10_000,
    ) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._embedding_engine = embedding_engine or EmbeddingEngine()
        self._max_items = max_items
        self._engine = create_async_engine(
            f"sqlite+aiosqlite:///{self._db_path}",
            echo=False,
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def initialize(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def store(
        self,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store a memory and return its ID."""
        embedding = self._embedding_engine.embed(content)

        async with self._session_factory() as session:
            now = datetime.now(timezone.utc)
            item = MemoryItem(
                content=content,
                embedding_json=json.dumps(embedding.tolist()),
                metadata_json=json.dumps(metadata or {}),
                tags=",".join(tags or []),
                created_at=now,
                accessed_at=now,
                access_count=0,
                relevance_score=1.0,
            )
            session.add(item)
            await session.commit()
            await session.refresh(item)

            # Prune oldest memories if over limit
            count = await session.scalar(select(func.count()).select_from(MemoryItem))
            if count and count > self._max_items:
                await self._prune(session, count - self._max_items)
                await session.commit()

            return int(item.id)

    async def forget(self, memory_id: int) -> bool:
        """Delete a specific memory by ID. Returns True if deleted."""
        async with self._session_factory() as session:
            item = await session.get(MemoryItem, memory_id)
            if not item:
                return False
            await session.delete(item)
            await session.commit()
            return True

    async def forget_by_content(self, content: str, threshold: float = 0.98) -> int:
        """Delete memories matching content semantically. Returns count deleted."""
        query_emb = self._embedding_engine.embed(content)
        items = await self._load_all(None)
        deleted = 0

        async with self._session_factory() as session:
            for item in items:
                sim = self._embedding_engine.similarity(query_emb, item.embedding)
                if sim >= threshold:
                    db_item = await session.get(MemoryItem, item.id)
                    if db_item:
                        await session.delete(db_item)
                        deleted += 1
            await session.commit()

        return deleted

    async def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.3,
        tags: list[str] | None = None,
    ) -> list[tuple[MemoryItem, float]]:
        """
        Retrieve memories ranked by composite relevance score.
        Returns (item, score) pairs sorted descending.
        """
        query_emb = self._embedding_engine.embed(query)
        items = await self._load_all(tags)

        if not items:
            return []

        embeddings = np.stack([item.embedding for item in items])
        cosine_scores = self._embedding_engine.batch_similarity(query_emb, embeddings)

        now_ts = time.time()
        max_access = max((item.access_count for item in items), default=1) or 1

        scored: list[tuple[MemoryItem, float]] = []
        for i, item in enumerate(items):
            cos = float(cosine_scores[i])
            if cos < threshold:
                continue

            age_days = (now_ts - item.created_at.timestamp()) / 86400.0
            recency = float(np.exp(-age_days / self.RECENCY_HALF_LIFE_DAYS))
            access_freq = float(np.log1p(item.access_count) / np.log1p(max_access))

            composite = (
                self.COSINE_WEIGHT * cos
                + self.RECENCY_WEIGHT * recency
                + self.ACCESS_WEIGHT * access_freq
            )
            scored.append((item, composite))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        # Update access metadata
        if top:
            async with self._session_factory() as session:
                now = datetime.now(timezone.utc)
                for item, _ in top:
                    db_item = await session.get(MemoryItem, item.id)
                    if db_item:
                        db_item.access_count += 1
                        db_item.accessed_at = now
                await session.commit()

        return top

    async def get_all(self, limit: int = 100, tags: list[str] | None = None) -> list[MemoryItem]:
        return await self._load_all(tags, limit=limit)

    async def count(self) -> int:
        async with self._session_factory() as session:
            result = await session.scalar(select(func.count()).select_from(MemoryItem))
            return int(result or 0)

    async def _load_all(
        self, tags: list[str] | None, limit: int | None = None
    ) -> list[MemoryItem]:
        async with self._session_factory() as session:
            stmt = select(MemoryItem).order_by(MemoryItem.accessed_at.desc())
            if limit:
                stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            items = list(result.scalars().all())

        if tags:
            tag_set = set(tags)
            items = [i for i in items if tag_set.intersection(i.tag_list)]

        return items

    async def _prune(self, session: AsyncSession, count: int) -> None:
        """Remove the oldest, least-accessed memories."""
        stmt = (
            select(MemoryItem)
            .order_by(MemoryItem.access_count.asc(), MemoryItem.accessed_at.asc())
            .limit(count)
        )
        result = await session.execute(stmt)
        for item in result.scalars():
            await session.delete(item)

    async def close(self) -> None:
        await self._engine.dispose()
