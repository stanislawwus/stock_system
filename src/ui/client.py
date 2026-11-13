"""httpx client for the API: the only way the Streamlit UI reads anything (AGENTS.md R13)."""

from __future__ import annotations

import os

import httpx

from src.schemas import AnalyzeResponse, Brief, ChatResponse, EventsResponse, SignalsResponse


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self.base_url = base_url or os.environ.get("API_URL", "http://localhost:8000/api/v1")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)

    def health(self) -> bool:
        root = self.base_url.rsplit("/api/", 1)[0]
        try:
            return httpx.get(f"{root}/health", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            return False

    def brief(self, day: str | None = None) -> Brief:
        r = self._http.get("/brief", params={"day": day} if day else None)
        r.raise_for_status()
        return Brief.model_validate(r.json())

    def signals(self, ticker: str, day: str | None = None) -> SignalsResponse:
        params = {"ticker": ticker, **({"day": day} if day else {})}
        r = self._http.get("/signals", params=params)
        r.raise_for_status()
        return SignalsResponse.model_validate(r.json())

    def events(self, ticker: str) -> EventsResponse:
        r = self._http.get(f"/events/{ticker}")
        r.raise_for_status()
        return EventsResponse.model_validate(r.json())

    def analyze(self, text: str, ticker: str | None = None) -> AnalyzeResponse:
        r = self._http.post("/analyze", json={"text": text, "ticker": ticker})
        r.raise_for_status()
        return AnalyzeResponse.model_validate(r.json())

    def chat(self, question: str) -> ChatResponse:
        r = self._http.post("/chat", json={"question": question})
        r.raise_for_status()
        return ChatResponse.model_validate(r.json())
