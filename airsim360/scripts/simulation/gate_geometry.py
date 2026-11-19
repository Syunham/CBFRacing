"""Oriented gate-frame boxes in planner world coordinates, in metres."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np


def gate_boxes(course, thickness=.12):
    boxes = []
    for i, (center, normal, size) in enumerate(zip(course["centers"], course["normals"], course["sizes"]), 1):
        nx, ny, _ = normal
        side = np.array([ny, -nx, 0.])  # right when looking along the normal
        up = np.array([0., 0., 1.])
        depth, width, height = size
        for label, offset, scale in [
            ("left", -side*(width+thickness)/2, [depth, thickness, height]),
            ("right", side*(width+thickness)/2, [depth, thickness, height]),
            ("top", up*(height+thickness)/2, [depth, width+2*thickness, thickness]),
            ("bottom", -up*(height+thickness)/2, [depth, width+2*thickness, thickness]),
        ]:
            boxes.append(dict(name=f"SplitS_G{i}_{label}", center=(center+offset).tolist(),
                              yaw=float(np.arctan2(ny,nx)), size=scale))
    return boxes
