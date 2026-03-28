"""Token-bucket rate limiter for API and scraping requests."""

import time
import threading
from functools import wraps


class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, requests_per_minute):
        self.interval = 60.0 / requests_per_minute
        self._lock = threading.Lock()
        self._last_request = 0.0

    def wait(self):
        """Block until the next request is allowed."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_request
            if elapsed < self.interval:
                time.sleep(self.interval - elapsed)
            self._last_request = time.monotonic()


def rate_limited(requests_per_minute):
    """Decorator to rate-limit a function."""
    limiter = RateLimiter(requests_per_minute)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait()
            return func(*args, **kwargs)
        return wrapper
    return decorator
