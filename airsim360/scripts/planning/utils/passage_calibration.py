"""Shared gate-plane features evaluated only at the existing gate events."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.paths import enter_project
    enter_project(__file__)

import numpy as np


def bend_features(centers,axes):
    incoming = centers - np.roll(centers,1,axis=0)
    incoming /= np.linalg.norm(incoming,axis=1,keepdims=True)
    bend = incoming - np.roll(incoming,-1,axis=0)
    return np.column_stack(((bend*axes).sum(axis=1),bend[:,2]))
