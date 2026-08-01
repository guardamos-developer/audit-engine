#!/usr/bin/env bash
# Run Layer1 tests using the project virtualenv (no global `python` required).
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt
python -m pytest tests/ -v "$@"
