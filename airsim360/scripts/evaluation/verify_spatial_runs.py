"""Verify the frozen model, sensing cadence, plant clearance and matched ablation."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import hashlib
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.evaluation.audit_spatial_flight import audit


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-perturbation", action="store_true")
    args = parser.parse_args()
    root = Path("reports/spatial_potential")
    model = root/"sensor12hz/best.pt"
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    rows = []
    for name, separation in (("validation08", .8), ("validation12", 1.2), ("validation13", 1.3)):
        folder = root/name
        cfg = read(folder/"experiment.json")
        assert cfg["checkpoint_sha256"] == digest and cfg["separation"] == separation and not cfg["field_off"]
        result = audit(folder/"flight")
        observations = [json.loads(line) for line in (folder/"sensor.jsonl").read_text().splitlines()]
        stamps = np.array([r["time"] for r in observations])
        ns = np.array([r["sensor_ns"] for r in observations])
        assert np.allclose(np.diff(stamps), .08) and np.all(np.diff(ns[:, 0, 0]) > 0)
        assert np.all(ns == ns[:, :1, :1])
        with np.load(folder/"flight/execution.npz") as f:
            t, x = f["time"], f["state"]
        assert stamps[0] == t[0] and -1e-8 <= t[-1]-stamps[-1] <= .08000001
        assert np.allclose(stamps, t[:4*len(stamps):4], atol=1e-7)
        assert np.isclose(np.linalg.norm(x[0, 0, :3]-x[0, 1, :3]), separation)
        rows.append(dict(run=name, separation=separation, synchronized_updates=len(stamps),
                         duration_s=float(t[-1]), **result))
    off = audit(root/"validation12_off/flight")
    off_cfg = read(root/"validation12_off/experiment.json")
    assert off_cfg["field_off"] and off_cfg["separation"] == 1.2 and off_cfg["corridor"]
    assert off_cfg["checkpoint_sha256"] == digest
    manifest = read(Path("reports/flight_count_comparison/manifest_8.json"))["episodes"]
    assert len({e["trajectory_sha256"] for e in manifest}) == len(manifest)
    assert all(hashlib.sha256((Path("reports/flight_count_comparison") / e["trajectory"]).read_bytes()).hexdigest() == e["trajectory_sha256"]
               for e in manifest)
    train = [e["episode_id"] for e in manifest if e["split"] == "train"]
    assert len(train) == 8
    report = dict(passed=all(r["passed"] for r in rows), model=str(model), sha256=digest,
                  train=train, test=[e["episode_id"] for e in manifest if e["split"] == "test"],
                  query="camera FRD (x,y,z), metres", sensing_hz=12.5, feedback_hz=50.,
                  physical_model="original 19-state numerical dynamics, AirSim online RGB-D",
                  trials=rows, field_off=off)
    if args.with_perturbation:
        shifted = root/"displaced_gate2"
        nominal = read(shifted/"flight/course.json")
        rendered = read(shifted/"rendered_course.json")
        delta = np.array(rendered["centers"])-nominal["centers"]
        expected = np.zeros_like(delta)
        expected[1] = .28*np.array(nominal["coordinate_axes"][1])
        np.testing.assert_allclose(delta, expected, atol=1e-7)
        matched = root/"displaced_gate2_passages_only"
        assert read(matched/"rendered_course.json") == rendered
        for folder in (shifted, matched):
            assert read(folder/"experiment.json")["checkpoint_sha256"] == digest
        ref = read(shifted/"flight/frontends.json")
        cmp = read(matched/"flight/frontends.json")
        for agent in (0, 1):
            first = lambda logs: next(r for r in logs if r["agent"] == agent and r["gate_index"] == 0)
            np.testing.assert_allclose(first(ref)["allocation"], first(cmp)["allocation"], atol=1e-8)
        result = audit(shifted/"flight")
        report["displaced_gate"] = dict(gate=2, offset_m=.28, online=result,
                                         fixed_passages=audit(matched/"flight"))
        report["passed"] = report["passed"] and result["passed"]
    (root/"validated.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
