#!/usr/bin/env python3
"""Launch the TPMS Heat Exchanger Optimizer GUI."""

import sys
import os
import tkinter as tk

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.gui import TPMSOptimizerGUI

if __name__ == "__main__":
    root = tk.Tk()
    app = TPMSOptimizerGUI(root)
    root.mainloop()

