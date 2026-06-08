#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python index.py --update
streamlit run app.py