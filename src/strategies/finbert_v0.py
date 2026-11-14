"""finbert_v0 (D-36): mean FinBERT score of the day's relevant, deduplicated news.

For decision day D (`view.as_of` = 09:00 ET of D) the window is (cutoff of the previous weekday,
as_of]: Monday's window covers the weekend. In it, the news tagged with the ticker are filtered
by relevance (symbol or alias mentioned), near-duplicates are removed keeping the earliest copy
(dedup looks `dedup.window_hours` further back so a copy of yesterday's story is not new), and
the score is the mean of the precomputed sentiment scores (P(pos) - P(neg)) of what is left.
Fewer than `min_articles` scored items -> 0 (cash). Long-only: score <= 0 means cash (R8).

Everything comes from the MarketView: news and sentiment rows with available_at <= as_of.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from src.marketview import MarketView
from src.marketview.clock import MARKET_TZ, cutoff_for
from src.nlp.relevance import config as nlp_config
from src.nlp.relevance import relevant_unique
from src.schemas import Signal
from src.strategies.base import make_signal, strategy_params

MAX_REASON_REFS = 5


def previous_cutoff(as_of: datetime) -> datetime:
    """Cutoff (09:00 ET) of the last weekday before the ET calendar day of `as_of`."""
    day = as_of.astimezone(MARKET_TZ).date() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return cutoff_for(day)


class FinbertV0:
    name = "finbert_v0"

    def __init__(self, method: str, min_articles: int, horizon_days: int | None) -> None:
        self.method = method
        self.min_articles = min_articles
        self.horizon_days = horizon_days

    @classmethod
    def from_config(cls) -> FinbertV0:
        return cls(**strategy_params(cls.name))

    def signal(self, ticker: str, view: MarketView) -> Signal:
        start = previous_cutoff(view.as_of)
        dedup_back = timedelta(hours=float(nlp_config()["dedup"]["window_hours"]))
        lookback = math.ceil((view.as_of - start + dedup_back) / timedelta(days=1))
        news = relevant_unique(view.news(ticker, lookback_days=lookback))
        today = [n for n in news if n.available_at > start]
        scores = {
            s.source_ref: s.score
            for s in view.sentiment(ticker, lookback_days=lookback, method=self.method)
        }
        used = [(n.id, scores[n.id]) for n in today if n.id in scores]
        if len(used) < self.min_articles:
            return make_signal(self.name, ticker, view, 0.0, f"news_n={len(used)}", "too_few")
        mean = sum(v for _, v in used) / len(used)
        refs = [i for i, _ in sorted(used, key=lambda x: -abs(x[1]))[:MAX_REASON_REFS]]
        return make_signal(
            self.name,
            ticker,
            view,
            mean,
            f"news_n={len(used)}",
            f"mean_{self.method}={mean:.3f}",
            *refs,
        )
