#!/bin/bash
# Launch GUI in Terminal.app (works around Cursor terminal limitations)

cd "$(dirname "$0")/.."

# Use osascript to open Terminal.app and run the GUI
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd '$(pwd)' && source venv/bin/activate 2>/dev/null || true && python3 scripts/run_gui.py"
end tell
EOF
