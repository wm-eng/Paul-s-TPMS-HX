-- AppleScript to launch GUI in Terminal.app
-- This works around Cursor's integrated terminal GUI limitations

tell application "Terminal"
    activate
    set currentTab to do script "cd '/Users/paulperera/Coding/TPMS-HX' && source venv/bin/activate && python scripts/run_gui.py"
end tell
