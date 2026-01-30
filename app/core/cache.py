"""
Embedding Cache - Persistent caching for vector embeddings.

This module provides disk-based caching for embeddings to avoid
redundant API calls and improve startup time.

Features:
- Content-hash based cache keys (stable across restarts)
- TTL-based expiration
- Atomic writes (no corruption on crash)
- LRU eviction when cache grows too large

Usage:
    from app.core.cache import get_embedding_cache
    
    cache = get_embedding_cache()
    
    # Check cache first
    embedding = cache.get(text)
    if embedding is None:
        embedding = compute_embedding(text)
        cache.set(text, embedding)
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A cached embedding entry."""
    embedding: list[float]
    created_at: float
    text_hash: str
    model: str
    
    def is_expired(self, ttl_hours: int) -> bool:
        """Check if entry has expired."""
        age_hours = (time.time() - self.created_at) / 3600
        return age_hours > ttl_hours
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "embedding": self.embedding,
            "created_at": self.created_at,
            "text_hash": self.text_hash,
            "model": self.model,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CacheEntry":
        """Create from dictionary."""
        return cls(
            embedding=data["embedding"],
            created_at=data["created_at"],
            text_hash=data["text_hash"],
            model=data["model"],
        )


class EmbeddingCache:
    """
    Persistent cache for vector embeddings.
    
    Stores embeddings on disk using content-hash keys for stability.
    Supports TTL-based expiration and automatic cleanup.
    
    Example:
        cache = EmbeddingCache()
        
        # Get or compute embedding
        text = "business rule about pricing"
        embedding = cache.get(text)
        if embedding is None:
            embedding = model.embed(text)
            cache.set(text, embedding)
    """
    
    _instance: Optional["EmbeddingCache"] = None
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_hours: int = 168,
        max_entries: int = 10000,
        model_name: str = "text-embedding-004",
    ) -> None:
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory for cache files (default from settings)
            ttl_hours: Time-to-live for entries in hours
            max_entries: Maximum number of entries before cleanup
            model_name: Embedding model name (for cache invalidation)
        """
        from app.config import get_settings
        
        settings = get_settings()
        
        self._cache_dir = cache_dir or settings.cache.embedding_cache_dir
        self._ttl_hours = ttl_hours or settings.cache.embedding_cache_ttl_hours
        self._max_entries = max_entries
        self._model_name = model_name
        self._enabled = settings.cache.enabled
        
        # In-memory cache for frequently accessed items
        self._memory_cache: dict[str, CacheEntry] = {}
        self._memory_cache_max = 1000
        
        # Ensure cache directory exists
        if self._enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Embedding cache initialized at {self._cache_dir}")
    
    @classmethod
    def get_instance(cls) -> "EmbeddingCache":
        """Get or create singleton cache instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None
    
    def _hash_text(self, text: str) -> str:
        """Generate stable hash for text content."""
        # Include model name in hash for cache invalidation on model change
        content = f"{self._model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def _get_cache_path(self, text_hash: str) -> Path:
        """Get file path for a cache entry."""
        # Use first 2 chars as subdirectory for filesystem efficiency
        subdir = text_hash[:2]
        return self._cache_dir / subdir / f"{text_hash}.json"
    
    def get(self, text: str) -> Optional[list[float]]:
        """
        Get embedding from cache.
        
        Args:
            text: Text that was embedded
            
        Returns:
            Cached embedding or None if not found/expired
        """
        if not self._enabled:
            return None
        
        text_hash = self._hash_text(text)
        
        # Check memory cache first
        if text_hash in self._memory_cache:
            entry = self._memory_cache[text_hash]
            if not entry.is_expired(self._ttl_hours):
                return entry.embedding
            else:
                del self._memory_cache[text_hash]
        
        # Check disk cache
        cache_path = self._get_cache_path(text_hash)
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            entry = CacheEntry.from_dict(data)
            
            # Check expiration
            if entry.is_expired(self._ttl_hours):
                cache_path.unlink(missing_ok=True)
                return None
            
            # Add to memory cache
            self._add_to_memory_cache(text_hash, entry)
            
            return entry.embedding
            
        except (json.JSONDecodeError, KeyError, OSError) as e:
            logger.warning(f"Failed to read cache entry {text_hash}: {e}")
            cache_path.unlink(missing_ok=True)
            return None
    
    def set(self, text: str, embedding: list[float]) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Original text
            embedding: Computed embedding vector
        """
        if not self._enabled:
            return
        
        text_hash = self._hash_text(text)
        
        entry = CacheEntry(
            embedding=embedding,
            created_at=time.time(),
            text_hash=text_hash,
            model=self._model_name,
        )
        
        # Store in memory cache
        self._add_to_memory_cache(text_hash, entry)
        
        # Store on disk (atomic write)
        cache_path = self._get_cache_path(text_hash)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temp file first, then rename (atomic)
        temp_path = cache_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(entry.to_dict(), f)
            temp_path.rename(cache_path)
        except OSError as e:
            logger.warning(f"Failed to write cache entry {text_hash}: {e}")
            temp_path.unlink(missing_ok=True)
    
    def get_batch(self, texts: list[str]) -> tuple[list[Optional[list[float]]], list[int]]:
        """
        Get multiple embeddings from cache.
        
        Args:
            texts: List of texts to look up
            
        Returns:
            Tuple of (embeddings, missing_indices) where embeddings contains
            None for cache misses, and missing_indices lists indices that
            need to be computed.
        """
        embeddings: list[Optional[list[float]]] = []
        missing_indices: list[int] = []
        
        for i, text in enumerate(texts):
            embedding = self.get(text)
            embeddings.append(embedding)
            if embedding is None:
                missing_indices.append(i)
        
        return embeddings, missing_indices
    
    def set_batch(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """
        Store multiple embeddings in cache.
        
        Args:
            texts: List of original texts
            embeddings: Corresponding embeddings
        """
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)
    
    def _add_to_memory_cache(self, text_hash: str, entry: CacheEntry) -> None:
        """Add entry to memory cache with LRU eviction."""
        if len(self._memory_cache) >= self._memory_cache_max:
            # Remove oldest entry (simple FIFO, not true LRU)
            oldest_key = next(iter(self._memory_cache))
            del self._memory_cache[oldest_key]
        
        self._memory_cache[text_hash] = entry
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()
        
        if self._cache_dir.exists():
            import shutil
            shutil.rmtree(self._cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Embedding cache cleared")
    
    def cleanup_expired(self) -> int:
        """
        Remove expired entries from disk cache.
        
        Returns:
            Number of entries removed
        """
        if not self._enabled or not self._cache_dir.exists():
            return 0
        
        removed = 0
        
        for subdir in self._cache_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            for cache_file in subdir.glob("*.json"):
                try:
                    with open(cache_file, 'r') as f:
                        data = json.load(f)
                    
                    entry = CacheEntry.from_dict(data)
                    if entry.is_expired(self._ttl_hours):
                        cache_file.unlink()
                        removed += 1
                        
                except (json.JSONDecodeError, KeyError, OSError):
                    # Remove corrupted entries
                    cache_file.unlink(missing_ok=True)
                    removed += 1
        
        logger.info(f"Cache cleanup removed {removed} expired entries")
        return removed
    
    def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with cache size, hit rate, etc.
        """
        stats = {
            "enabled": self._enabled,
            "memory_entries": len(self._memory_cache),
            "memory_max": self._memory_cache_max,
            "ttl_hours": self._ttl_hours,
            "cache_dir": str(self._cache_dir),
        }
        
        if self._enabled and self._cache_dir.exists():
            disk_entries = sum(
                len(list(subdir.glob("*.json")))
                for subdir in self._cache_dir.iterdir()
                if subdir.is_dir()
            )
            stats["disk_entries"] = disk_entries
            
            # Estimate disk usage
            total_size = sum(
                f.stat().st_size
                for subdir in self._cache_dir.iterdir()
                if subdir.is_dir()
                for f in subdir.glob("*.json")
            )
            stats["disk_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        return stats


def get_embedding_cache() -> EmbeddingCache:
    """Get the global embedding cache instance."""
    return EmbeddingCache.get_instance()
