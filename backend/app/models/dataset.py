"""SQLAlchemy ORM model for uploaded datasets."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base


class Dataset(Base):
    """Metadata for a single uploaded dataset.

    One row is created every time a user uploads a CSV file. The actual file
    lives on disk (see ``dataset_path``); only lightweight metadata is stored
    in the database.
    """

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    upload_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    rows: Mapped[int] = mapped_column(Integer, nullable=False)
    columns: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    dataset_path: Mapped[str] = mapped_column(String(512), nullable=False)

    def __repr__(self) -> str:  # pragma: no cover - debugging convenience
        return (
            f"<Dataset id={self.id} filename={self.filename!r} "
            f"rows={self.rows} columns={self.columns}>"
        )
