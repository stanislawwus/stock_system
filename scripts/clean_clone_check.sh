#!/bin/sh
# Gate 3 (D-48): clone the repo into a temp dir, `make up`, and smoke-test the running stack.
#   sh scripts/clean_clone_check.sh [repo-path-or-url]      (POSIX sh; runs in Git Bash)
# Checks (exit 0 only if every one passes):
#   boot     /api/v1/health, /docs, the UI health endpoint, ports bound to 127.0.0.1 only
#   S1       /analyze answers with FinBERT (method "finbert"), not the lexicon fallback
#   S2       no LLM access (no keys in the container): /chat and /signals still answer, with
#            citations, "coincided in time" and the disclaimer
#   S3       unknown ticker: /chat refuses, /signals answers 404
#   S4       test-period date: /signals answers 403, /chat refuses (R3)
#   S5       daily LLM budget exhausted (LLM_DAILY_BUDGET=0): FinBERT, signals and template chat
#            answers still work; the UI shows "Degraded mode" (checked with Streamlit's AppTest
#            inside the ui container against the live API; without the budget problem it must not)
# Only the committed HEAD is cloned: commit before running. No secret is read or printed.
set -u

SRC=${1:-$(cd "$(dirname "$0")/.." && pwd)}
API=http://127.0.0.1:8000
UI=http://127.0.0.1:8501
TMP=$(mktemp -d 2>/dev/null || mktemp -d -t cleanclone)
CLONE="$TMP/clone"
PASS=0
FAIL=0
STARTED=0

say() { printf '%s\n' "$*"; }
ok() { PASS=$((PASS + 1)); say "  PASS $1"; }
bad() { FAIL=$((FAIL + 1)); say "  FAIL $1"; }

cleanup() {
  if [ "$STARTED" = 1 ] && [ -d "$CLONE" ]; then
    say "== shutting down"
    (cd "$CLONE" && docker compose down --remove-orphans >/dev/null 2>&1)
  fi
  rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

# check NAME COMMAND... : pass when the command succeeds
check() {
  name=$1
  shift
  if "$@" >/dev/null 2>&1; then ok "$name"; else bad "$name"; fi
}

# has TEXT PATTERN : fixed-string match of PATTERN in TEXT
has() { printf '%s' "$1" | grep -qF -- "$2"; }

http_code() { curl -s -o /dev/null -m 60 -w '%{http_code}' "$@"; }

wait_health() {
  i=0
  while [ "$i" -lt "${1:-60}" ]; do
    [ "$(http_code "$API/api/v1/health")" = 200 ] && return 0
    i=$((i + 1))
    sleep 2
  done
  return 1
}

ui_degraded() { # EXPECT=true|false: does the Streamlit app render the degraded banner?
  (cd "$CLONE" && docker compose exec -T ui python -c "
import sys
from streamlit.testing.v1 import AppTest
at = AppTest.from_file('src/ui/app.py', default_timeout=90).run()
degraded = any('Degraded mode' in w.value for w in at.warning)
print('degraded=%s exceptions=%d' % (degraded, len(at.exception)))
sys.exit(0 if (degraded == ('$1' == 'true') and not at.exception) else 1)
")
}

post() { curl -s -m 120 -X POST "$API/api/v1/$1" -H 'Content-Type: application/json' -d "$2"; }

say "== clone $SRC"
if ! git clone --quiet "$SRC" "$CLONE"; then say "FAIL: git clone"; exit 1; fi
say "   $(cd "$CLONE" && git log --oneline -1)"
if [ "$(http_code "$API/api/v1/health")" = 200 ] || [ "$(http_code "$UI/_stcore/health")" = 200 ]; then
  say "FAIL: ports 8000/8501 are already in use (run 'make down' first)"
  STARTED=0
  exit 1
fi

say "== make up (builds the image: FinBERT weights are baked in; the first build is slow)"
STARTED=1
if ! (cd "$CLONE" && make up); then say "FAIL: make up"; exit 1; fi

say "== boot"
check "health answers" wait_health 30
check "/docs is served" test "$(http_code "$API/docs")" = 200
check "UI health endpoint" test "$(curl -s -m 30 "$UI/_stcore/health")" = ok
check "UI page is served" test "$(http_code "$UI/")" = 200
ports=$(cd "$CLONE" && docker compose ps --format '{{.Publishers}}' 2>/dev/null)
if has "$ports" "0.0.0.0" || has "$ports" "[::]"; then bad "ports bound to 127.0.0.1 only"; else ok "ports bound to 127.0.0.1 only"; fi
check "UI shows no degraded banner with budget left" ui_degraded false

say "== S1 FinBERT, not the lexicon"
body=$(post analyze '{"text": "Boeing shares plunge after FAA grounds jets", "ticker": "BA"}')
if has "$body" '"method":"finbert"'; then ok "analyze uses FinBERT"; else bad "analyze uses FinBERT: $(printf '%s' "$body" | cut -c1-200)"; fi
if has "$body" 'not investment advice'; then ok "analyze carries the disclaimer"; else bad "analyze carries the disclaimer"; fi

say "== S2 no LLM access (the container holds no API key)"
body=$(post chat '{"question": "Why did NVDA move on 2026-08-27?"}')
has "$body" '"method":"rules"' && has "$body" 'coincided in time' && has "$body" '"citations":[{' \
  && has "$body" 'not investment advice' && ok "chat answers from the template with citations" \
  || bad "chat answers from the template with citations: $(printf '%s' "$body" | cut -c1-200)"
body=$(curl -s -m 60 "$API/api/v1/signals?ticker=AAPL&as_of=2026-08-14")
has "$body" '"signals":[{' && has "$body" 'not investment advice' && ok "signals answer" || bad "signals answer"

say "== S3 unknown ticker"
body=$(post chat '{"question": "Why did XYZQ move on 2026-07-15?"}')
has "$body" '"refused":true' && has "$body" 'unknown_ticker' && ok "chat refuses an unknown ticker" \
  || bad "chat refuses an unknown ticker"
check "signals answers 404 for an unknown ticker" test "$(http_code "$API/api/v1/signals?ticker=XYZQ")" = 404

say "== S4 test period and guaranteed profit"
check "signals refuse a test-period date (403)" test "$(http_code "$API/api/v1/signals?ticker=AAPL&as_of=2026-10-01")" = 403
body=$(post chat '{"question": "Which stock is guaranteed to double my money?"}')
has "$body" '"refused":true' && has "$body" 'guaranteed_profit' && ok "chat refuses a guaranteed-profit request" \
  || bad "chat refuses a guaranteed-profit request"

say "== S5 daily LLM budget exhausted (LLM_DAILY_BUDGET=0)"
if (cd "$CLONE" && LLM_DAILY_BUDGET=0 docker compose up -d --wait >/dev/null 2>&1); then
  ok "stack restarts with the budget at 0"
else
  bad "stack restarts with the budget at 0"
fi
check "health after restart" wait_health 30
body=$(curl -s -m 30 "$API/api/v1/models")
has "$body" '"daily_budget":0' && ok "/models reports the exhausted budget" || bad "/models reports the exhausted budget"
body=$(post analyze '{"text": "Boeing shares plunge after FAA grounds jets", "ticker": "BA"}')
has "$body" '"method":"finbert"' && ok "FinBERT still works" || bad "FinBERT still works"
body=$(post chat '{"question": "Why did NVDA move on 2026-08-27?"}')
has "$body" '"method":"rules"' && has "$body" 'coincided in time' && ok "template chat answer still works" \
  || bad "template chat answer still works"
check "signals still work" test "$(http_code "$API/api/v1/signals?ticker=AAPL&as_of=2026-08-14")" = 200
check "UI shows the degraded banner" ui_degraded true

say "== result: $PASS passed, $FAIL failed"
[ "$FAIL" = 0 ] && say "GATE 3: PASS" || say "GATE 3: FAIL"
[ "$FAIL" = 0 ]
