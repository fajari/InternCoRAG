from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import requests

from config import LLM_ENABLED, LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_REQUEST_TIMEOUT

_OLLAMA_AVAILABLE: bool | None = None


class OllamaClient:
    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = OLLAMA_MODEL,
        timeout: float = OLLAMA_REQUEST_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(LLM_ENABLED and LLM_PROVIDER.lower() == "ollama" and self.is_available())

    def is_available(self) -> bool:
        global _OLLAMA_AVAILABLE
        if _OLLAMA_AVAILABLE is not None:
            return _OLLAMA_AVAILABLE
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=(0.4, 0.8))
            _OLLAMA_AVAILABLE = response.status_code < 500
        except Exception:
            _OLLAMA_AVAILABLE = False
        return _OLLAMA_AVAILABLE

    async def generate(self, prompt: str, system: str = "") -> str:
        if not self.enabled:
            return ""
        return await asyncio.to_thread(self.generate_sync, prompt, system)

    def generate_sync(self, prompt: str, system: str = "") -> str:
        if not self.enabled:
            return ""

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "top_p": 0.2,
            },
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(1.0, self.timeout),
            )
            response.raise_for_status()
            data = response.json()
            return str(data.get("response", "")).strip()
        except Exception:
            return ""


def parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return {}

    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def parse_json_list(text: str) -> list[dict[str, Any]]:
    if not text:
        return []

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        if isinstance(data, dict):
            events = data.get("events")
            if isinstance(events, list):
                return [item for item in events if isinstance(item, dict)]
    except Exception:
        pass

    match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
    except Exception:
        return []
