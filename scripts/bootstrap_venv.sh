#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"
INSTALL_APP="${INSTALL_APP_REQUIREMENTS:-0}"

python -m venv --system-site-packages "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-core.txt
python -m pip install -e . --no-deps

if [[ "$INSTALL_APP" == "1" ]]; then
  python -m pip install -r requirements-app.txt
fi

echo
echo "Virtual environment created at $VENV_DIR"
echo "Activate it with: source $VENV_DIR/bin/activate"
echo "Then verify with: python scripts/check_env.py"
if [[ "$INSTALL_APP" == "1" ]]; then
  echo "Streamlit support installed."
else
  echo "Streamlit not installed. Set INSTALL_APP_REQUIREMENTS=1 if you want the app dependencies."
fi
