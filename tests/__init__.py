"""Test suite.

Several tests deliberately exercise failure paths that log warnings (a missing
Piper voice, an unreadable sidecar, a jammed servo). Those messages are the
expected behaviour, not test output, so logging is muted here to keep the
results readable.
"""

import logging

logging.disable(logging.CRITICAL)
