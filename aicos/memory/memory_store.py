"""
Persistent memory store backed by SQLite (dev) or PostgreSQL+pgvector (prod).

Search path selection:
  PostgreSQL + pgvector column → fast ANN search via <=> operator (scales to millions)
  Otherwise                    → full-scan JSON + NumPy (fine up to ~10k items)

Scoring formula (both paths):
    score = cosine_sim * 0.60 + recency_decay * 0.25 + access_freq * 0.15
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlalchemy import String, Float, Integer, Text, DateTime, select, func
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import text

from aicos.core.database import build_engine
from aicos.core.logging import get_logger
from aicos.memory.embeddings import EmbeddingEngine

log = get_logger("memory.store")


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
    Async memory store.

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
        database_url: str,
        embedding_engine: EmbeddingEngine | None = None,
        max_items: int = 10_000,
    ) -> None:
        self._embedding_engine = embedding_engine or EmbeddingEngine()
        self._max_items = max_items
        self._engine = build_engine(database_url)
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )
        self._pgvector = False  # set to True in initialize() if column found

    async def initialize(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        self._pgvector = await self._detect_pgvector()
        if self._pgvector:
            log.info("MemoryStore: pgvector search enabled")

    async def _detect_pgvector(self) -> bool:
        """Return True if the embedding_vec column exists (added by migration 002)."""
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT embedding_vec FROM memories LIMIT 0"))
            return True
        except Exception:
            return False

    async def store(
        self,
        content: str,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store a memory and return its ID."""
        embedding = self._embedding_engine.embed(content)
        vec_str = str(embedding.tolist())

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
            await session.flush()  # get id without committing
            item_id = int(item.id)

            if self._pgvector:
                await session.execute(
                    text(
                        "UPDATE memories SET embedding_vec = CAST(:vec AS vector)"
                        " WHERE id = :id"
                    ),
                    {"vec": vec_str, "id": item_id},
                )

            await session.commit()

            count = await session.scalar(select(func.count()).select_from(MemoryItem))
            if count and count > self._max_items:
                await self._prune(session, count - self._max_items)
                await session.commit()

            return item_id

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

        Uses pgvector ANN index when available — loads only top candidates,
        not the full table. Falls back to full-scan for SQLite or when the
        pgvector extension is not installed.
        """
        query_emb = self._embedding_engine.embed(query)

        if self._pgvector:
            top = await self._search_pgvector(query_emb, top_k, threshold, tags)
        else:
            top = await self._search_json(query_emb, top_k, threshold, tags)

        if top:
            await self._update_access(top)
        return top

    async def _search_pgvector(
        self,
        query_emb: np.ndarray,
        top_k: int,
        threshold: float,
        tags: list[str] | None,
    ) -> list[tuple[MemoryItem, float]]:
        """ANN search via the <=> cosine-distance operator."""
        vec_str = str(query_emb.tolist())
        fetch_limit = top_k * 5  # oversample for composite re-scoring

        async with self._session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT id, 1 - (embedding_vec <=> CAST(:qv AS vector)) AS cosine_sim
                    FROM memories
                    WHERE embedding_vec IS NOT NULL
                      AND 1 - (embedding_vec <=> CAST(:qv AS vector)) >= :threshold
                    ORDER BY embedding_vec <=> CAST(:qv AS vector)
                    LIMIT :lim
                """),
                {"qv": vec_str, "threshold": threshold, "lim": fetch_limit},
            )
            id_cos: list[tuple[int, float]] = [(r[0], float(r[1])) for r in result.fetchall()]

        if not id_cos:
            return []

        cosine_by_id = {item_id: cos for item_id, cos in id_cos}
        ids = list(cosine_by_id)

        async with self._session_factory() as session:
            rows = await session.execute(
                select(MemoryItem).where(MemoryItem.id.in_(ids))
            )
            items = list(rows.scalars().all())

        if tags:
            tag_set = set(tags)
            items = [i for i in items if tag_set.intersection(i.tag_list)]

        return self._composite_score(items, cosine_by_id, threshold, top_k)

    async def _search_json(
        self,
        query_emb: np.ndarray,
        top_k: int,
        threshold: float,
        tags: list[str] | None,
    ) -> list[tuple[MemoryItem, float]]:
        """Full-scan similarity via NumPy (SQLite / no pgvector)."""
        items = await self._load_all(tags)
        if not items:
            return []

        embeddings = np.stack([item.embedding for item in items])
        cosine_scores = self._embedding_engine.batch_similarity(query_emb, embeddings)
        cosine_by_id = {item.id: float(cosine_scores[i]) for i, item in enumerate(items)}

        return self._composite_score(items, cosine_by_id, threshold, top_k)

    def _composite_score(
        self,
        items: list[MemoryItem],
        cosine_by_id: dict[int, float],
        threshold: float,
        top_k: int,
    ) -> list[tuple[MemoryItem, float]]:
        now_ts = time.time()
        max_access = max((item.access_count for item in items), default=1) or 1

        scored: list[tuple[MemoryItem, float]] = []
        for item in items:
            cos = cosine_by_id.get(item.id, 0.0)
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
        return scored[:top_k]

    async def _update_access(self, results: list[tuple[MemoryItem, float]]) -> None:
        async with self._session_factory() as session:
            now = datetime.now(timezone.utc)
            for item, _ in results:
                db_item = await session.get(MemoryItem, item.id)
                if db_item:
                    db_item.access_count += 1
                    db_item.accessed_at = now
            await session.commit()

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
