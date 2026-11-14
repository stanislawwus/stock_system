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

from src.marketview import MarketView
from src.nlp.daily import previous_cutoff, window_scores
from src.schemas import Signal
from src.strategies.base import make_signal, strategy_params

__all__ = ["FinbertV0", "previous_cutoff"]

MAX_REASON_REFS = 5


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
        used = window_scores(ticker, view, self.method)
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
