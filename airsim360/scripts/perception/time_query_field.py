"""Frozen four-frame RGB-D predictor and its exact MPCC query decoder."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import casadi as ca
import numpy as np
import torch
import torch.nn.functional as F

from scripts.training.train_rgbd_field import RGBDField, image_history


def load_rgbd(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if saved.get("projected") or saved["query_frame"] != "current_camera_FRD":
        raise ValueError("Expected the original camera-frame RGBDField checkpoint")
    model = RGBDField(wide=saved.get("wide",False)).eval()
    model.native_wide = saved.get("wide",False)
    model.load_state_dict(saved["model"])
    return model


@torch.inference_mode()
def encode_views(model, rgb, depth):
    # Match the 256x128 encoder: average RGB, select a real range sample
    # per output pixel to preserve invalid returns and object boundaries.
    views = torch.stack([image_history(rgb[:,i],depth[:,i],np.ones(4,bool)) for i in (0,1)])
    if getattr(model, "native_wide", False):
        if views.shape[-2:] != (256,512):
            raise ValueError("Wide checkpoint requires native 512x256 panoramas")
        return model.encode(views)
    frames = views.flatten(0,1)
    frames = torch.cat([F.interpolate(frames[:,:3],size=(128,256),mode="area"),
                        F.interpolate(frames[:,3:],size=(128,256),mode="nearest-exact")],dim=1)
    return model.encode(frames.reshape(2,4,5,128,256))


def symbolic_query(checkpoint):
    net = load_rgbd(checkpoint).decoder
    q = ca.SX.sym("camera_frd",3)
    tau = ca.SX.sym("lead_s")
    bias = ca.SX.sym("image_bias",96)
    u = bias + ca.DM(net[0].weight.detach().numpy()[:,64:]) @ ca.vertcat(q/3,tau/1.6)
    for layer in (net[2],net[4]):
        u = u * (.5 + .5*ca.tanh(u/2))
        u = ca.DM(layer.weight.detach().numpy()) @ u + ca.DM(layer.bias.detach().numpy())
    return ca.Function("rgbd_time_query",[q,tau,bias],[.5+.5*ca.tanh(u/2)],{"never_inline":True})
