"""Shared utilities for pipeline stages."""

import logging
import time

import requests

logger = logging.getLogger(__name__)


def post_with_retry(
    url: str,
    json_body: dict,
    timeout: int,
    max_attempts: int = 3,
) -> requests.Response:
    """POST with exponential-backoff retry on connection/timeout errors.

    Retries only transient network failures (ConnectionError, Timeout).
    HTTP 4xx/5xx raised by raise_for_status() are NOT retried — they
    indicate semantic failures the caller should surface immediately.
    """
    for attempt in range(max_attempts):
        try:
            r = requests.post(url, json=json_body, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            if attempt == max_attempts - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                "Ollama call failed (attempt %d/%d), retrying in %ds: %s",
                attempt + 1, max_attempts, wait, exc,
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")
