"""
Pytest auto-discovers this file at the project root before collecting tests.
It adds src/ to sys.path so test modules can do `from model import SimpleCNN`
etc., matching how the scripts themselves import each other.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))