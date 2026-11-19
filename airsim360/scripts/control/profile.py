"""Controller settings shared by the racing frontends."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

from scripts.paths import FLIGHT_ASSETS
import json


def baseline_options(validate, name, corridor=False, *, replan_period=None):
    base = json.loads((FLIGHT_ASSETS/"numerical_baseline.json").read_text())
    period = 0. if corridor else base["replan_period"]
    if replan_period is not None:
        period = replan_period
    options = validate.parse_args(["--name", name, "--cbf", "learned", "--replan-period", str(period)])
    for key in vars(options):
        if key in base:
            setattr(options, key, base[key])
    if replan_period is not None:
        options.replan_period = period
    return options


def configure_racing_tracker(options):
    options.phase_first_gate_aperture = False
    options.phase_passage_half = .075
    options.phase_vertical_half = .10
    options.gate_prediction_margin = .25
    options.reward_policy = "fixed"
    options.progress_start = options.progress = 360.
    options.turn_reward = 0.
    options.contour_weight, options.lag_weight = 2800., 3500.
    options.reference_speed_weight, options.reference_speed_extra = 400., -.3
    options.velocity_contour_weight = 36.
