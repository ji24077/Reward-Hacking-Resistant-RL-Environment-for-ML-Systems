import sys
import os

# Ensure src/ and root are importable from pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.dirname(__file__))
