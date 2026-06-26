"""Tiny thread-safe LRU for rendered cube bytes.

Keyed by request string; values are the gzip NIfTI payloads. A small cache is
enough because zoom interaction revisits the same/neighbouring cubes.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional


class LRUBytesCache:
    def __init__(self, capacity: int = 256) -> None:
        self.capacity = capacity
        self._d: "OrderedDict[str, bytes]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            if key not in self._d:
                return None
            self._d.move_to_end(key)
            return self._d[key]

    def put(self, key: str, value: bytes) -> None:
        with self._lock:
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.capacity:
                self._d.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._d)
