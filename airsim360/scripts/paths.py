"""Locations resolved within this AirSim360 installation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLIGHT_ASSETS = ROOT / "assets" / "flight"


def enter_project(entry):
    """Resolve project-relative inputs and avoid shadowing stdlib modules."""
    import os
    import sys
    if len(sys.path) > 1 and Path(sys.path[1]).resolve() == Path(entry).resolve().parent:
        del sys.path[1]
    os.chdir(ROOT)
