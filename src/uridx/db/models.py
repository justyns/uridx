from datetime import datetime
from typing import Optional

from sqlmodel import Field, Relationship, SQLModel


class Tag(SQLModel, table=True):
    item_id: int = Field(foreign_key="item.id", primary_key=True)
    tag: str = Field(primary_key=True, index=True)

    item: "Item" = Relationship(back_populates="tags")


class Chunk(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    item_id: int = Field(foreign_key="item.id", index=True)
    chunk_key: Optional[str] = None
    chunk_index: Optional[int] = None
    text: str
    meta: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    item: "Item" = Relationship(back_populates="chunks")


class Item(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    source_uri: str = Field(unique=True, index=True)
    title: Optional[str] = None
    source_type: Optional[str] = Field(default=None, index=True)
    context: Optional[str] = None
    expires_at: Optional[datetime] = Field(default=None, index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    chunks: list[Chunk] = Relationship(back_populates="item")
    tags: list[Tag] = Relationship(back_populates="item")


class Setting(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)
