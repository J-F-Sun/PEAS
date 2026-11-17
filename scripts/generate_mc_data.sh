#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "${JULIA:-julia}" --project="$ROOT/baselines/monte_carlo" "$ROOT/baselines/monte_carlo/scripts/ams_mod.jl" "$@"
