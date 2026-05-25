"""conversationId 별 세션 상태 (멀티턴 컨텍스트) — 메모리 캐시 + TTL.

generator.py 의 state 구조: {"summary": str, "last_q": str, "last_a": str}.
TTL 만료 시 새 컨텍스트로 시작.

운영 시 multi-instance 라면 Redis 등으로 교체 가능.
단일 프로세스 인메모리면 충분.
"""

from __future__ import annotations

import threading
import time
from typing import Dict

from config import SESSION_TTL_SECONDS


def _empty_state() -> Dict[str, str]:
    return {"summary": "", "last_q": "", "last_a": ""}


class SessionStore:
    def __init__(self, ttl_seconds: int = SESSION_TTL_SECONDS) -> None:
        self._store: Dict[str, Dict[str, str]] = {}
        self._expiry: Dict[str, float] = {}
        self._ttl = ttl_seconds
        self._lock = threading.Lock()

    def get(self, conv_id: str) -> Dict[str, str]:
        with self._lock:
            self._evict_expired()
            return dict(self._store.get(conv_id, _empty_state()))

    def set(self, conv_id: str, state: Dict[str, str]) -> None:
        with self._lock:
            self._store[conv_id] = dict(state)
            self._expiry[conv_id] = time.time() + self._ttl

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [cid for cid, t in self._expiry.items() if t < now]
        for cid in expired:
            self._store.pop(cid, None)
            self._expiry.pop(cid, None)
