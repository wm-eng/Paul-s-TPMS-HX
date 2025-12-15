#!/usr/bin/env python3
"""Launch the TPMS Heat Exchanger Optimizer GUI."""

import sys
import os

# macOS-specific fixes for Tkinter
if sys.platform == 'darwin':
    # Set environment variables to help with macOS GUI issues
    os.environ['TK_SILENCE_DEPRECATION'] = '1'
    
    # Check if we're in a GUI-capable environment
    display = os.environ.get('DISPLAY')
    if not display:
        # Try to set a default display (though this may not work in all cases)
        pass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

if __name__ == "__main__":
    try:
        import tkinter as tk
    except ImportError:
        print("Error: tkinter not available. Please install Python with tkinter support.")
        print("On macOS, tkinter should come with Python. Try: python3 -m tkinter")
        sys.exit(1)
    
    try:
        from hxopt.gui import TPMSOptimizerGUI
    except ImportError as e:
        print(f"Error importing GUI module: {e}")
        print("Make sure you're in the project directory and dependencies are installed.")
        sys.exit(1)
    
    try:
        # Create root window with minimal initialization first
        root = tk.Tk()
        
        # Set macOS-specific window properties before creating the app
        if sys.platform == 'darwin':
            try:
                # Set up proper app bundle behavior
                root.createcommand('tk::mac::ReopenApplication', lambda: root.deiconify())
            except Exception:
                pass  # Ignore if this fails
        
        # Now create the application
        app = TPMSOptimizerGUI(root)
        
        # Start the main loop
        root.mainloop()
        
    except RuntimeError as e:
        if "no display name" in str(e).lower() or "cannot open display" in str(e).lower():
            print("Error: No display available. Cannot launch GUI.")
            print("\nTroubleshooting:")
            print("1. Run from Terminal.app (not Cursor's integrated terminal)")
            print("2. Or run: open -a Terminal scripts/run_gui.py")
            print("3. Or use the macOS launcher: ./scripts/run_gui_macos.sh")
        else:
            print(f"Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching GUI: {e}")
        print("\nTroubleshooting tips:")
        print("1. Try running from Terminal.app instead of Cursor's integrated terminal")
        print("2. Run: ./scripts/run_gui_macos.sh")
        print("3. Or: pythonw scripts/run_gui.py (if pythonw is available)")
        print("4. Check tkinter: python3 -m tkinter")
        import traceback
        traceback.print_exc()
        sys.exit(1)

