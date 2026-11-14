-- DuckDB tables. Every table served by MarketView has available_at TIMESTAMPTZ NOT NULL (R2, D-24).
-- impact_labels is the only table with forward-looking values; MarketView never serves it.

CREATE TABLE IF NOT EXISTS price_bars (
  ticker       VARCHAR NOT NULL,
  bar_date     DATE NOT NULL,
  open         DOUBLE NOT NULL,
  high         DOUBLE NOT NULL,
  low          DOUBLE NOT NULL,
  close        DOUBLE NOT NULL,
  volume       DOUBLE NOT NULL,
  source       VARCHAR NOT NULL,
  available_at TIMESTAMPTZ NOT NULL,          -- bar close time
  PRIMARY KEY (ticker, bar_date)
);

CREATE TABLE IF NOT EXISTS news (
  id           VARCHAR NOT NULL,              -- sha1(url + headline); one article may tag several tickers
  ticker       VARCHAR NOT NULL,
  published_at TIMESTAMPTZ NOT NULL,
  headline     VARCHAR NOT NULL,
  summary      VARCHAR NOT NULL DEFAULT '',
  url          VARCHAR NOT NULL,
  source       VARCHAR NOT NULL,
  available_at TIMESTAMPTZ NOT NULL,          -- = published_at
  PRIMARY KEY (id, ticker)
);

CREATE TABLE IF NOT EXISTS filings (
  accession_no VARCHAR PRIMARY KEY,
  ticker       VARCHAR NOT NULL,
  cik          VARCHAR NOT NULL,
  form         VARCHAR NOT NULL,              -- 8-K | 10-Q | 10-K | 4
  items        VARCHAR[] NOT NULL,            -- 8-K item numbers from the EDGAR index
  accepted_at  TIMESTAMPTZ NOT NULL,
  filed_date   DATE NOT NULL,
  url          VARCHAR NOT NULL,
  text_path    VARCHAR,
  available_at TIMESTAMPTZ NOT NULL           -- = accepted_at
);

CREATE TABLE IF NOT EXISTS facts (
  ticker       VARCHAR NOT NULL,
  name         VARCHAR NOT NULL,              -- revenue | eps_diluted | net_income
  value        DOUBLE NOT NULL,
  unit         VARCHAR NOT NULL,
  period_end   DATE NOT NULL,
  fiscal_period VARCHAR NOT NULL,
  method       VARCHAR NOT NULL,              -- xbrl | llm | regex
  source_ref   VARCHAR NOT NULL,
  grounding    JSON,                          -- Span for extracted facts
  available_at TIMESTAMPTZ NOT NULL,          -- xbrl: acceptance of the carrying 10-Q/10-K (D-30); extracted: source's
  PRIMARY KEY (ticker, name, period_end, method, source_ref)
);

CREATE TABLE IF NOT EXISTS sentiment (
  source_ref   VARCHAR NOT NULL,
  ticker       VARCHAR NOT NULL,
  label        INTEGER NOT NULL,              -- -1 | 0 | 1
  score        DOUBLE NOT NULL,
  method       VARCHAR NOT NULL,
  available_at TIMESTAMPTZ NOT NULL,          -- = source's available_at (D-16)
  PRIMARY KEY (source_ref, ticker, method)    -- one article can be tagged with several tickers (D-36)
);

CREATE TABLE IF NOT EXISTS events (
  source_ref   VARCHAR NOT NULL,
  ticker       VARCHAR NOT NULL,
  labels       VARCHAR[] NOT NULL,
  confidence   DOUBLE NOT NULL,
  method       VARCHAR NOT NULL,
  available_at TIMESTAMPTZ NOT NULL,          -- = source's available_at (D-16)
  PRIMARY KEY (source_ref, method)
);

-- Forward-looking labels (docs/03_eval_plan.md). Never served by MarketView.
CREATE TABLE IF NOT EXISTS impact_labels (
  source_ref   VARCHAR NOT NULL,
  ticker       VARCHAR NOT NULL,
  event_available_at TIMESTAMPTZ NOT NULL,
  decision_day DATE NOT NULL,                 -- D: first trading day with cutoff(D) >= event_available_at
  gap_ar       DOUBLE,                        -- (open(D)/close(D-1) - 1) - same for SPY: priced in at entry
  drift_ar     DOUBLE,                        -- (close(D+1)/open(D) - 1) - same for SPY: tradable target
  PRIMARY KEY (source_ref)
);
