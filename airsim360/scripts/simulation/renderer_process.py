"""AirSim sensor worker process shared by the depth and RGB-D pipelines."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class RendererProcess:
    def start_renderer(self, checkpoint_geometry, folder, gate_offset, depth_only=False):
        command = [str(ROOT/".venv/python.exe"), "-u", "-m", "scripts.simulation.spatial_render_worker",
                   str(folder/"sensor.jsonl"), "--resolution", str(self.width), str(self.height)]
        if checkpoint_geometry:
            (folder/"vehicle_geometry.json").write_text(json.dumps(checkpoint_geometry, indent=2))
            command += ["--scale", *map(str, checkpoint_geometry["scale_xyz"])]
        if gate_offset:
            command += ["--gate-offset", *map(str, gate_offset)]
        if depth_only:
            command.append("--depth-only")
        self.worker = subprocess.Popen(command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=ROOT)
        if self.worker.stdout.readline().strip() != b"READY":
            self.close()
            raise RuntimeError("Online AirSim renderer startup failed")

    def close(self):
        self.worker.stdin.close()
        try:
            self.worker.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.worker.kill()
            self.worker.wait()
        self.worker.stdout.close()
        self.log.close()
