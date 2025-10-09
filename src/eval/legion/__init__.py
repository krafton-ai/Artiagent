"""
LEGION Evaluation Package

This package contains LEGION-specific evaluation tools and utilities for
artifact detection and localization evaluation using pre-generated responses.
"""

# Add parent directory to path for imports when this package is imported
import os
import sys

# Add the eval directory to the Python path so we can import from it
_eval_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _eval_dir not in sys.path:
    sys.path.insert(0, _eval_dir)
