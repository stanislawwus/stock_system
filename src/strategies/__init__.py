"""Strategy arena: every strategy implements `signal(ticker, view) -> Signal`, where `view` is a
MarketView at the 09:00 ET cutoff of the decision day (it carries `as_of`). Long-only: a score
<= 0 means cash. Strategies: buy_and_hold, momentum (12-1), mean_reversion (D-35).
Aggregation (Hedge, D-17) will live here too.
"""
