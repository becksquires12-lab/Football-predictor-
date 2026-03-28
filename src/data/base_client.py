"""Abstract base class for HTTP clients with retry logic."""

from abc import ABC, abstractmethod

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.utils.logger import get_logger


class BaseClient(ABC):
    """Base HTTP client with session management and retries."""

    def __init__(self, base_url, headers=None, retries=3, backoff_factor=0.5):
        self.base_url = base_url.rstrip("/")
        self.logger = get_logger(self.__class__.__name__)
        self.session = requests.Session()

        if headers:
            self.session.headers.update(headers)

        retry_strategy = Retry(
            total=retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get(self, endpoint, params=None):
        """Make a GET request and return parsed JSON."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        self.logger.debug(f"GET {url} params={params}")

        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def _get_html(self, endpoint, params=None):
        """Make a GET request and return raw HTML text."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        self.logger.debug(f"GET (html) {url}")

        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.text

    @abstractmethod
    def get_matches(self, league, season):
        """Fetch matches for a given league and season."""
        pass
