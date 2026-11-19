"""Re-solve and audit a saved five-gate Fast-fly planning problem."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.planning.fast_fly.guarded_fast_fly import PassagePhase
from scripts.planning.utils.local_planner import save


def main():
    """Re-solve one five-gate problem without the renderer or the MPCC loop."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("flight", type=Path, help="Flight directory with course.json, planner_config.json and plans/")
    parser.add_argument("--plan", default="000_agent0_event0.npz", help="Plan filename inside plans/")
    parser.add_argument("--output", type=Path, help="Save the solved plan to this NPZ file")
    args = parser.parse_args()
    config = json.loads((args.flight / "planner_config.json").read_text())
    course = json.loads((args.flight / "course.json").read_text())
    # A cold, single solve has no prior SQP state. Build derivatives from the
    # recorded model, independently of the machine that recorded the flight.
    config = {key: value for key, value in config.items() if not key.endswith("_library")}
    config.update(phase_online_sqp=False, phase_cell_function=True)
    planner = PassagePhase(config, course)
    with np.load(args.flight / "plans" / args.plan) as inputs:
        planner.event_index = int(Path(args.plan).stem.rsplit("event", 1)[1])
        state = inputs["audited_initial_state"]
        planner.initial_forces = inputs["initial_forces"] if planner.initial_force_weight else None
        gates, normals, half, times = (inputs[key] for key in ("gates", "normals", "half", "event_durations_input"))
        plan = planner.solve(state, gates, normals, half, times)
        plan["allocation"] = inputs["allocation"]
    if not plan["audit"]["passed"]:
        raise RuntimeError(f"Plan failed the dense dynamics and gate audit: {plan['audit']}")
    if args.output:
        save(args.output, plan)
    print(json.dumps(dict(durations_s=plan["durations"].tolist(), audit=plan["audit"]), indent=2))


if __name__ == "__main__":
    main()
