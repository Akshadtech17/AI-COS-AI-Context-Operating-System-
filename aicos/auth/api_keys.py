"""
SQLite-backed API key store.

Keys are stored as SHA-256 hashes — the plaintext is shown exactly once at
creation and never persisted. This mirrors the GitHub PAT / Stripe key design.

Key format:  aicos-{8-char-prefix}-{40-hex-chars}
Store format: sha256(plaintext) — never the raw key
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import Boolean, Float, Integer, String, select
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from aicos.core.database import build_engine


class _Base(DeclarativeBase):
    pass


class _APIKeyRow(_Base):
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(200))
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    prefix: Mapped[str] = mapped_column(String(20))
    created_at: Mapped[float] = mapped_column(Float)
    last_used_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    scopes: Mapped[str] = mapped_column(String(500), default="chat,memory")


@dataclass
class APIKey:
    id: int
    name: str
    prefix: str
    created_at: datetime
    last_used_at: datetime | None
    is_active: bool
    scopes: list[str]


class APIKeyStore:
    """
    Create, validate, list, and revoke gateway API keys.

    All keys are stored hashed. The plaintext is returned once at creation;
    if lost, the key must be revoked and replaced.
    """

    def __init__(self, database_url: str) -> None:
        self._engine = build_engine(database_url)
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)

    async def initialize(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)

    async def create_key(
        self,
        name: str,
        scopes: list[str] | None = None,
    ) -> tuple[str, APIKey]:
        """
        Create a new API key.

        Returns (plaintext_key, APIKey). The plaintext is NEVER stored —
        show it to the user exactly once. If lost, revoke and regenerate.
        """
        plaintext = f"aicos-{secrets.token_urlsafe(6)}-{secrets.token_hex(20)}"
        prefix = plaintext[:16]
        key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
        scope_str = ",".join(scopes or ["chat", "memory"])
        now = time.time()

        async with self._sessions() as session:
            row = _APIKeyRow(
                name=name,
                key_hash=key_hash,
                prefix=prefix,
                created_at=now,
                is_active=True,
                scopes=scope_str,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)

        api_key = APIKey(
            id=row.id,
            name=row.name,
            prefix=row.prefix,
            created_at=datetime.fromtimestamp(now, tz=UTC),
            last_used_at=None,
            is_active=True,
            scopes=scope_str.split(","),
        )
        return plaintext, api_key

    async def validate(self, plaintext: str) -> APIKey | None:
        """Validate a key and update last_used_at. Returns None if invalid."""
        key_hash = hashlib.sha256(plaintext.encode()).hexdigest()
        async with self._sessions() as session:
            result = await session.execute(
                select(_APIKeyRow).where(
                    _APIKeyRow.key_hash == key_hash,
                    _APIKeyRow.is_active.is_(True),
                )
            )
            row = result.scalar_one_or_none()
            if not row:
                return None
            row.last_used_at = time.time()
            await session.commit()
            return _row_to_key(row)

    async def revoke(self, key_id: int) -> bool:
        """Revoke a key by ID. Returns True if found and revoked."""
        async with self._sessions() as session:
            result = await session.execute(select(_APIKeyRow).where(_APIKeyRow.id == key_id))
            row = result.scalar_one_or_none()
            if not row:
                return False
            row.is_active = False
            await session.commit()
            return True

    async def list_keys(self) -> list[APIKey]:
        """List all active API keys (hashes not exposed)."""
        async with self._sessions() as session:
            result = await session.execute(
                select(_APIKeyRow)
                .where(_APIKeyRow.is_active.is_(True))
                .order_by(_APIKeyRow.created_at.desc())
            )
            return [_row_to_key(row) for row in result.scalars().all()]

    async def close(self) -> None:
        await self._engine.dispose()


def _row_to_key(row: _APIKeyRow) -> APIKey:
    return APIKey(
        id=row.id,
        name=row.name,
        prefix=row.prefix,
        created_at=datetime.fromtimestamp(row.created_at, tz=UTC),
        last_used_at=(
            datetime.fromtimestamp(row.last_used_at, tz=UTC) if row.last_used_at else None
        ),
        is_active=row.is_active,
        scopes=row.scopes.split(","),
    )
