"""Connectivity detection.

The draft script called ``urllib.request.urlopen('https://1.1.1.1', timeout=2)``
once per recognised utterance. That has two problems: it negotiates TLS against
a bare IP (the certificate check can fail even with a healthy connection, so the
robot decides it is offline while it is not), and it puts a blocking round trip
on the critical path between hearing a sentence and answering it.

This module instead opens a bare TCP socket to a DNS port and caches the answer
for a short window.
"""

from __future__ import annotations

import logging
import socket
import time

from .config import NetworkConfig

log = logging.getLogger(__name__)


class ConnectivityMonitor:
    """Answers "are we online?" without stalling the speech loop."""

    def __init__(self, config: NetworkConfig, clock=time.monotonic) -> None:
        self._config = config
        self._clock = clock
        self._cached: bool | None = None
        self._checked_at: float = 0.0

    def is_online(self, *, force: bool = False) -> bool:
        """Return cached connectivity, re-probing once the TTL has expired."""
        now = self._clock()
        fresh = (
            self._cached is not None
            and (now - self._checked_at) < self._config.cache_ttl_s
        )
        if fresh and not force:
            return self._cached  # type: ignore[return-value]

        self._cached = self._probe()
        self._checked_at = now
        return self._cached

    def invalidate(self) -> None:
        """Drop the cache so the next call re-probes."""
        self._cached = None
        self._checked_at = 0.0

    def _probe(self) -> bool:
        try:
            with socket.create_connection(
                (self._config.probe_host, self._config.probe_port),
                timeout=self._config.timeout_s,
            ):
                return True
        except OSError as exc:
            log.debug("connectivity probe failed: %s", exc)
            return False
