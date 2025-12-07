"""Shared OpenAI client helpers that stay compatible with openai>=1.10.

The official SDK removed the legacy ``proxies`` keyword argument, so we
provide a thin wrapper that builds ``httpx`` clients when proxy variables
are present while keeping the public signature free of unsupported
parameters. Both sync and async variants are exposed for reuse across the
project.
"""

from __future__ import annotations

import os
from typing import Optional

import httpx
from openai import AsyncOpenAI, OpenAI

# We intentionally reuse standard proxy environment variables instead of a
# ``proxies`` argument to avoid TypeErrors on openai>=1.10.
_PROXY_ENV_KEYS = ("OPENAI_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY")


def _get_proxy_from_env() -> Optional[str]:
    for key in _PROXY_ENV_KEYS:
        value = os.getenv(key)
        if value:
            return value
    return None


def _build_httpx_client(timeout: float = 60) -> httpx.Client:
    proxy = _get_proxy_from_env()
    transport = httpx.HTTPTransport(retries=3)
    if proxy:
        return httpx.Client(proxies=proxy, transport=transport, timeout=timeout)
    return httpx.Client(transport=transport, timeout=timeout)


def _build_async_httpx_client(timeout: float = 60) -> httpx.AsyncClient:
    proxy = _get_proxy_from_env()
    transport = httpx.AsyncHTTPTransport(retries=3)
    if proxy:
        return httpx.AsyncClient(proxies=proxy, transport=transport, timeout=timeout)
    return httpx.AsyncClient(transport=transport, timeout=timeout)


def build_openai_client(api_key: str | None = None, base_url: str | None = None, timeout: float = 60) -> OpenAI:
    client_kwargs = {"api_key": api_key, "base_url": base_url, "http_client": _build_httpx_client(timeout)}
    # Drop None values so the SDK can apply its defaults.
    filtered_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}
    return OpenAI(**filtered_kwargs)


def build_async_openai_client(api_key: str | None = None, base_url: str | None = None, timeout: float = 60) -> AsyncOpenAI:
    client_kwargs = {"api_key": api_key, "base_url": base_url, "http_client": _build_async_httpx_client(timeout)}
    filtered_kwargs = {k: v for k, v in client_kwargs.items() if v is not None}
    return AsyncOpenAI(**filtered_kwargs)
