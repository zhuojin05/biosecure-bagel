#!/usr/bin/env bash
# Master launcher for overnight Modal batch jobs — biosecurity screening only.
#
# Backgrounds all jobs concurrently; Modal handles GPU allocation per call.
# Wall time dominated by PP screening jobs (~3-7 hours each).
#
# Job 1: 4-signal validation (~5 min, 56 calls)
# Job 2: PP screening — 3 canonical templates (~2-5 hr, ~8325 calls)
# Job 3: Integration tests (~25 min, ~20 calls)
# Job 4: PP screening — extended template panel (~3-7 hr, ~13440 calls)
# Job 5: False-positive stress test (~10 min, 100 calls)
#
# Usage:
#   chmod +x scripts/run_overnight.sh
#   ./scripts/run_overnight.sh
#
# Prerequisites:
#   modal token set --token-id <NEW_ID> --token-secret <NEW_SECRET>

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$ROOT/logs/overnight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG"

echo "=== Starting overnight batch at $(date) ===" | tee "$LOG/master.log"
echo "Log directory: $LOG" | tee -a "$LOG/master.log"
echo "" | tee -a "$LOG/master.log"

# Job 1: 4-signal validation (fast ~5 min, ESMFold + ESM2 over 28 sequences)
echo "[$(date +%H:%M:%S)] Launching Job 1: 4-signal validation ..." | tee -a "$LOG/master.log"
(cd "$ROOT" && uv run python "$ROOT/scripts/overnight/validate_full_signals.py") \
    2>&1 | tee "$LOG/validate_full.log" &
JOB1_PID=$!

# Job 2: PP screening batch (~2-5 hours, ~8325 ESMFold calls)
echo "[$(date +%H:%M:%S)] Launching Job 2: PP screening batch ..." | tee -a "$LOG/master.log"
(cd "$ROOT" && uv run python "$ROOT/scripts/overnight/pp_screening_batch.py") \
    2>&1 | tee "$LOG/pp_screening.log" &
JOB2_PID=$!

# Job 3: Integration tests with real Modal GPUs (~25 min)
# cd to repo root so uv resolves the correct project and bagel is importable
echo "[$(date +%H:%M:%S)] Launching Job 3: integration tests ..." | tee -a "$LOG/master.log"
(cd "$ROOT" && uv run pytest tests/integration_tests/ --oracles modal -q) \
    2>&1 | tee "$LOG/integration_tests.log" &
JOB3_PID=$!

# Job 4: PP screening — extended template panel (~3-7 hr, ~13440 ESMFold calls)
echo "[$(date +%H:%M:%S)] Launching Job 4: PP extended template panel ..." | tee -a "$LOG/master.log"
(cd "$ROOT" && uv run python "$ROOT/scripts/overnight/pp_extended_templates.py") \
    2>&1 | tee "$LOG/pp_extended.log" &
JOB4_PID=$!

# Job 5: False-positive stress test (~10 min, 100 oracle calls)
echo "[$(date +%H:%M:%S)] Launching Job 5: false-positive stress test ..." | tee -a "$LOG/master.log"
(cd "$ROOT" && uv run python "$ROOT/scripts/overnight/stress_test_triage.py") \
    2>&1 | tee "$LOG/stress_test.log" &
JOB5_PID=$!

echo "" | tee -a "$LOG/master.log"
echo "All jobs launched. PIDs:" | tee -a "$LOG/master.log"
echo "  Job 1 (validate_full):        $JOB1_PID" | tee -a "$LOG/master.log"
echo "  Job 2 (pp_screening):         $JOB2_PID" | tee -a "$LOG/master.log"
echo "  Job 3 (integration_tests):    $JOB3_PID" | tee -a "$LOG/master.log"
echo "  Job 4 (pp_extended):          $JOB4_PID" | tee -a "$LOG/master.log"
echo "  Job 5 (stress_test):          $JOB5_PID" | tee -a "$LOG/master.log"
echo "" | tee -a "$LOG/master.log"
echo "Monitor with: tail -f $LOG/*.log" | tee -a "$LOG/master.log"

# Wait for all background jobs
wait

echo "" | tee -a "$LOG/master.log"
echo "=== All jobs finished at $(date) ===" | tee -a "$LOG/master.log"

# Print a quick summary of which logs are non-empty
echo "" | tee -a "$LOG/master.log"
echo "--- Log sizes ---" | tee -a "$LOG/master.log"
ls -lh "$LOG"/*.log | tee -a "$LOG/master.log"
