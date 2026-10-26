"""Command-line compatibility entry point for the package dataset generator."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from practo_support.dataset import APPOINTMENTS, generate_appointments, main

__all__ = ["APPOINTMENTS", "generate_appointments"]


if __name__ == "__main__":
    main()
