"""
Base declarative class and async engine setup for RAG database.
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all RAG database models."""
    pass
