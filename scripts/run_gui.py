#!/usr/bin/env python3
"""Launch the TPMS Heat Exchanger Optimizer GUI."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hxopt.gui import main

if __name__ == "__main__":
    main()

