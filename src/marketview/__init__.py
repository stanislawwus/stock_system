"""MarketView(as_of): the only read path for historical data. Returns rows with available_at <=
as_of. Readers open DuckDB read_only. See AGENTS.md rule 3.
"""
