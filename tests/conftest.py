"""
Shared pytest fixtures for ptt_gui tests.
"""

import sys
import os

# Make sure the src/ directory is on sys.path so imports work without install.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
