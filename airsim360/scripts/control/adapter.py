"""Prepare run artifacts and select the project-local vehicle model."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json

from scripts.paths import ROOT
from scripts.control import closed_loop
from scripts.simulation.dynamics import planner_configuration, select_vehicle


def controller(adapter, *, vehicle=None):
    select_vehicle(vehicle)
    adapter.mkdir(parents=True)
    (adapter / "config.json").write_text(json.dumps(planner_configuration(), indent=2))
    return closed_loop
