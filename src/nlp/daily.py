"""The decision-day news window shared by finbert_v0, the impact features and the aggregator.

For decision day D (`view.as_of` = 09:00 ET of D) the window is (cutoff of the previous weekday,
as_of]: Monday's window covers the weekend. News tagged with the ticker are filtered by relevance
and near-duplicates removed keeping the earliest copy (dedup looks `dedup.window_hours` further
back so a copy of yesterday's story is not new). Everything comes from the MarketView (R2).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from src.marketview import MarketView
from src.marketview.clock import MARKET_TZ, cutoff_for
from src.nlp.relevance import config as nlp_config
from src.nlp.relevance import relevant_unique


def previous_cutoff(as_of: datetime) -> datetime:
    """Cutoff (09:00 ET) of the last weekday before the ET calendar day of `as_of`."""
    day = as_of.astimezone(MARKET_TZ).date() - timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return cutoff_for(day)


def window_scores(ticker: str, view: MarketView, method: str) -> list[tuple[str, float]]:
    """(news id, sentiment score) of the relevant, deduplicated, scored news in D's window."""
    start = previous_cutoff(view.as_of)
    dedup_back = timedelta(hours=float(nlp_config()["dedup"]["window_hours"]))
    lookback = math.ceil((view.as_of - start + dedup_back) / timedelta(days=1))
    news = relevant_unique(view.news(ticker, lookback_days=lookback))
    today = [n for n in news if n.available_at > start]
    scores = {
        s.source_ref: s.score for s in view.sentiment(ticker, lookback_days=lookback, method=method)
    }
    return [(n.id, scores[n.id]) for n in today if n.id in scores]
