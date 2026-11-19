"""Check cached queries, spatial derivatives and the actual patched MPCC objective on a held-out flight."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import torch

from scripts.control.adapter import ROOT, controller
from scripts.perception.live_vision import LiveVision
from scripts.perception.implicit_rgbd import HISTORIES, HORIZON, history_indices, load_field
from scripts.training.train_implicit_rgbd import observation
from scripts.training.train_rgbd_field import Windows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--manifest", type=Path, default=Path("captures/cf21_512_double_50hz/manifest.json"))
    args = cli.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    ep = Windows(args.manifest, "test", 1, .25, lazy_windows=True).episodes[0]
    times, states, rgb, depth = ep[:4]
    os.environ.update(RGBD_IMPLICIT="1", ABLATION_MODE="threat", THREAT_WEIGHT="1",
                      RISK_TEMPORAL_DECAY="0", RGBD_DIRECTIONAL="0", RGBD_MIXTURE="0")
    os.environ.pop("RGBD_TIME_CHECKPOINT", None)
    with TemporaryDirectory(dir=args.output) as temp:
        controller(Path(temp)/"adapter")
        from scripts.control.rti_problem import RecoveryMPCC, DT
        import casadi as ca
        class Replay(LiveVision):
            def start_renderer(self, **kwargs):
                pass

            def capture(self, stamp, unused_states):
                i = np.abs(times-stamp).argmin()
                return rgb[i].copy(), depth[i].copy()

        spatial = ROOT/"reports/spatial_potential/crazyflie21_512/best.pt"
        live = Replay(spatial, args.output, implicit_checkpoint=args.checkpoint)
        model, history = load_field(args.checkpoint)
        encoding_ms, query_ms = [], []
        k = 46
        for i in range(k-18, k+1):
            live.observe(float(times[i]), states[i])
        field = live.implicit_fields[0]
        # Isolate single-agent new-image preprocessing and encoding from simulator I/O.
        for i in range(k+1, k+22):
            begin = perf_counter()
            field.observe(float(times[i]), states[i, 0], rgb[i, 0].copy(), depth[i, 0].copy())
            encoding_ms.append((perf_counter()-begin)*1000)
        k = i
        # Supply the same final image to the spatial branch and the other agent.
        live.observe(float(times[k]), states[k])
        live.control_stamp = float(times[k])+.02
        node_times = live.control_stamp+np.arange(21)*DT
        ids = np.abs(times[:, None]-node_times).argmin(axis=0)
        nodes = states[ids, 0, :3].copy()
        nodes += np.random.default_rng(36).uniform(-.23, .23, nodes.shape)
        for _ in range(24):
            begin = perf_counter()
            logits, grad = field.linearize(nodes, node_times)
            query_ms.append((perf_counter()-begin)*1000)
        inputs = [v[None] for v in observation(ep, k, 0, history)]
        tau = torch.tensor((node_times-times[k]).clip(0, HORIZON), dtype=torch.float32)[None, :, None]
        q = torch.tensor(nodes[None], dtype=torch.float32)
        with torch.no_grad():
            offline = model(*inputs, q, tau)[0].numpy()
        np.testing.assert_allclose(logits, offline, atol=2e-5, rtol=1e-5)

        # Central differences with double precision distinguish real pixel-boundary
        # changes from subtraction noise. Compare to actual float32 MPCC gradients.
        model.double()
        inputs = [v.double() for v in inputs]
        with torch.no_grad():
            frames = [model.encode_frame(*(v[:, j] for v in inputs)) for j in range(4)]
            q64, t64 = q.double(), tau.double()
            eps = 1e-6
            numeric = []
            for axis in torch.eye(3, dtype=torch.float64):
                plus = model.query_logits(frames, q64+eps*axis, t64)
                minus = model.query_logits(frames, q64-eps*axis, t64)
                numeric.append(((plus-minus)/(2*eps))[0].numpy())
        numeric = np.stack(numeric, axis=-1)
        np.testing.assert_allclose(grad, numeric, rtol=.025, atol=.003)
        if not np.any(abs(grad) > 1e-5):
            raise AssertionError("Degenerate test gradients")

        # Use LiveVision.install and the exact solver objective with other costs zero.
        problem = RecoveryMPCC(0, n=20, cbf_mode="none", build_only=True)
        state_nodes = states[ids, 0].copy()
        state_nodes[:, :3] = nodes
        problem.seed(state_nodes, np.zeros((20, 5)))
        od = np.zeros((21, 44))
        live.install(problem, od)
        live.log.close()
        np.testing.assert_allclose(od[:, 21:24], grad)
        np.testing.assert_allclose(od[:, 24], logits)
        problem.data(od)
        objective = ca.Function("audit", [problem.z, problem.p],
                                [problem.objective, ca.gradient(problem.objective, problem.z)])
        z = np.r_[state_nodes.ravel(), problem.U.ravel()]
        def check_objective(data):
            params = np.r_[state_nodes[0], data.ravel(), np.zeros(14)]
            value, derivative = objective(z, params)
            delta = nodes-data[:, 16:19]
            distance = np.sqrt((delta**2).sum(1)+1e-10)
            fixed = (data[:, 19]-distance)/data[:, 20]
            dynamic = data[:, 24]
            score = .5+.5*np.tanh(np.maximum(fixed, dynamic)/2)
            df = np.where((fixed > dynamic)[:, None], -delta/(distance*data[:, 20])[:, None], grad)
            expected_gradient = (2*score**2*(1-score))[:, None]*df
            np.testing.assert_allclose(float(value), score@score, atol=1e-7)
            symbolic_grad = np.array(derivative).ravel()[:21*19].reshape(21, 19)[:, :3]
            np.testing.assert_allclose(symbolic_grad, expected_gradient, atol=1e-6)
            return int((dynamic > fixed).sum())
        active = check_objective(problem.od)
        # Force the gate surface to dominate on one node and peer risk on another.
        controlled = problem.od.copy()
        controlled[:2, 16:19] = nodes[:2]+[[.15, 0, 0], [15, 0, 0]]
        controlled[:2, 19:21] = [.5, .02]
        check_objective(controlled)

        def latency(values):
            return dict(zip(("p50", "p95"), np.percentile(values[1:], [50, 95])))
        report = dict(nodes=len(nodes), agent=0, history=history, frame=int(k),
                      history_ids=history_indices(times, times[k], HISTORIES[history]).tolist(),
                      observation_age_s=.02, lead_cap_s=HORIZON, active_dynamic_nodes=active,
                      max_cached_logit_error=float(abs(logits-offline).max()),
                      max_world_gradient_error=float(abs(grad-numeric).max()),
                      symbolic_cost_and_gradient_checked=True,
                      single_core_new_frame_ms=latency(encoding_ms),
                      single_core_cached_21_logits_and_grad_ms=latency(query_ms))
        (args.output/"check.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report))


if __name__ == "__main__":
    main()
