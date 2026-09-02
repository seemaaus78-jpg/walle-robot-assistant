"""Connectivity caching, using a fake clock so the tests take no real time."""

import unittest

from walle.config import NetworkConfig
from walle.net import ConnectivityMonitor


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CountingMonitor(ConnectivityMonitor):
    """Replaces the socket probe with a counter."""

    def __init__(self, config, clock, result=True):
        super().__init__(config, clock=clock)
        self.result = result
        self.probes = 0

    def _probe(self) -> bool:
        self.probes += 1
        return self.result


class CachingTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.config = NetworkConfig(cache_ttl_s=30.0)
        self.monitor = CountingMonitor(self.config, self.clock)

    def test_repeated_calls_reuse_the_cache(self):
        # The draft probed the network once per utterance, adding a blocking
        # round trip between hearing a sentence and answering it.
        for _ in range(10):
            self.assertTrue(self.monitor.is_online())
        self.assertEqual(self.monitor.probes, 1)

    def test_cache_expires_after_the_ttl(self):
        self.monitor.is_online()
        self.clock.advance(29.0)
        self.monitor.is_online()
        self.assertEqual(self.monitor.probes, 1)
        self.clock.advance(2.0)
        self.monitor.is_online()
        self.assertEqual(self.monitor.probes, 2)

    def test_force_bypasses_the_cache(self):
        self.monitor.is_online()
        self.monitor.is_online(force=True)
        self.assertEqual(self.monitor.probes, 2)

    def test_invalidate_forces_the_next_call_to_probe(self):
        self.monitor.is_online()
        self.monitor.invalidate()
        self.monitor.is_online()
        self.assertEqual(self.monitor.probes, 2)

    def test_offline_result_is_cached_too(self):
        monitor = CountingMonitor(self.config, self.clock, result=False)
        self.assertFalse(monitor.is_online())
        self.assertFalse(monitor.is_online())
        self.assertEqual(monitor.probes, 1)


class ProbeTests(unittest.TestCase):
    def test_unroutable_address_reports_offline_without_raising(self):
        # 203.0.113.0/24 is TEST-NET-3 and is guaranteed not to be routed.
        config = NetworkConfig(probe_host="203.0.113.1", probe_port=9, timeout_s=0.25)
        self.assertFalse(ConnectivityMonitor(config).is_online())


if __name__ == "__main__":
    unittest.main()
