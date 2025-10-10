"""Quick test script to verify dh5merge GUI functionality."""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dh5cli.dh5merge import main

if __name__ == "__main__":
    # This will open the GUI since no args are provided
    main()
