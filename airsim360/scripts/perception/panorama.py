"""Unit camera-frame rays at ERP pixel centres."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

from functools import lru_cache
import numpy as np


@lru_cache(maxsize=2)
def spherical_rays(height, width):
    v, u = np.indices((height, width), dtype=np.float32) + .5
    lon, lat = (u / width - .5) * (2*np.pi), (v / height - .5) * np.pi
    return np.stack((np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)), -1)
