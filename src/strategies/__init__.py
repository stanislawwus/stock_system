"""Strategy arena: every strategy implements signal(ticker, as_of, view) -> Signal. Long-only:
value <= 0 means cash. Aggregation (Hedge) lives here too.
"""
