#!/bin/bash
# macOS-specific launcher for the GUI
# This script helps work around macOS Tkinter issues

cd "$(dirname "$0")/.."

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Try pythonw first (better for GUI apps on macOS)
if command -v pythonw &> /dev/null; then
    pythonw scripts/run_gui.py
else
    # Fall back to python3
    python3 scripts/run_gui.py
fi
