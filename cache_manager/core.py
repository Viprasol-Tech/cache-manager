"""
Cache manager - Cache and reuse LLM responses.

Hash-based caching with TTL support and statistics.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, Any


@dataclass
class CacheEntry:
    """Single cache entry."""
    key: str
    value: str
    created_at: datetime = field(default_factory=datetime.now)
    ttl_seconds: int = 3600  # 1 hour default
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl_seconds

    def hit(self):
        """Record cache hit."""
        self.hits += 1


class CacheManager:
    """Manage response cache."""

    def __init__(self, max_entries: int = 1000):
        """Initialize cache."""
        self.cache: Dict[str, CacheEntry] = {}
        self.max_entries = max_entries
        self.total_hits = 0
        self.total_misses = 0

    @staticmethod
    def hash_prompt(prompt: str, model: str = "") -> str:
        """Create cache key from prompt."""
        combined = f"{model}:{prompt}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get(self, key: str) -> Optional[str]:
        """Get from cache."""
        if key not in self.cache:
            self.total_misses += 1
            return None

        entry = self.cache[key]

        if entry.is_expired:
            del self.cache[key]
            self.total_misses += 1
            return None

        entry.hit()
        self.total_hits += 1
        return entry.value

    def set(self, key: str, value: str, ttl_seconds: int = 3600) -> None:
        """Set cache entry."""
        # Check size limit
        if len(self.cache) >= self.max_entries:
            self._evict_oldest()

        self.cache[key] = CacheEntry(key=key, value=value, ttl_seconds=ttl_seconds)

    def get_or_set(self, key: str, callback, ttl_seconds: int = 3600) -> str:
        """Get from cache or compute."""
        cached = self.get(key)
        if cached:
            return cached

        value = callback()
        self.set(key, value, ttl_seconds)
        return value

    def clear_expired(self) -> int:
        """Remove expired entries."""
        expired_keys = [k for k, v in self.cache.items() if v.is_expired]
        for key in expired_keys:
            del self.cache[key]
        return len(expired_keys)

    def _evict_oldest(self) -> None:
        """Remove oldest entry."""
        if not self.cache:
            return

        oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k].created_at)
        del self.cache[oldest_key]

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        self.clear_expired()

        total_requests = self.total_hits + self.total_misses
        hit_rate = (self.total_hits / total_requests * 100) if total_requests > 0 else 0

        return {
            "entries": len(self.cache),
            "max_entries": self.max_entries,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "hit_rate": round(hit_rate, 2),
            "size_percentage": round((len(self.cache) / self.max_entries) * 100, 1),
        }

    def clear(self) -> int:
        """Clear all entries."""
        count = len(self.cache)
        self.cache.clear()
        return count


class PromptCache:
    """Cache for prompt-response pairs."""

    def __init__(self):
        """Initialize prompt cache."""
        self.manager = CacheManager()

    def cache_response(self, prompt: str, response: str, model: str = "", ttl: int = 3600) -> str:
        """Cache a response."""
        key = CacheManager.hash_prompt(prompt, model)
        self.manager.set(key, response, ttl)
        return key

    def get_cached_response(self, prompt: str, model: str = "") -> Optional[str]:
        """Get cached response."""
        key = CacheManager.hash_prompt(prompt, model)
        return self.manager.get(key)

    def batch_cache(self, prompts: list, responses: list, model: str = "") -> int:
        """Cache multiple responses."""
        if len(prompts) != len(responses):
            raise ValueError("Prompts and responses must have same length")

        for prompt, response in zip(prompts, responses):
            self.cache_response(prompt, response, model)

        return len(prompts)


def create_cache(max_entries: int = 1000) -> CacheManager:
    """Create cache."""
    return CacheManager(max_entries)


def get_cached(cache: CacheManager, key: str) -> Optional[str]:
    """Quick get."""
    return cache.get(key)


def process(data: str, **kwargs) -> str:
    """Process function."""
    return "Cache manager ready"


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage LLM response cache")
    parser.add_argument("-m", "--max-entries", type=int, default=1000)
    parser.add_argument("--stats", action="store_true")

    args = parser.parse_args()

    cache = CacheManager(max_entries=args.max_entries)

    if args.stats:
        stats = cache.get_stats()
        print("Cache Statistics:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
    else:
        print(f"Cache manager ready (max {args.max_entries} entries)")


if __name__ == "__main__":
    main()
