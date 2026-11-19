"""CF2X brushed rigid-body profile for the scaled Crazyflie 2.1 renderer."""

if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.paths import enter_project
    enter_project(__file__)

import math


def vehicle_parameters():
    # gym-pybullet-drones/assets/cf2x.urdf; thrust = kf * RPM**2.
    mass, gravity = 0.027, 9.81
    kf, km = 3.16e-10, 7.94e-12
    hover_force = mass * gravity / 4
    max_force = 2.25 * hover_force
    # The shared model has constant body drag. Evaluate the CF2X rotor-speed
    # drag coefficient at hover and convert RPM to rad/s as in BaseAviary.
    rotor_speed_sum = 4 * math.sqrt(hover_force / kf) * 2 * math.pi / 60
    return dict(mass=mass, gravity=gravity, arm_length=0.0397,
                inertia=[1.4e-5, 1.4e-5, 2.17e-5], torque_coefficient=km / kf,
                thrust_min=0., thrust_max=max_force,
                # Preserve the baseline's 35 ms full-range slew envelope.
                thrust_rate_max=max_force / 0.035,
                linear_drag_kg_s=[c * rotor_speed_sum for c in (9.1785e-7, 9.1785e-7, 10.311e-7)])
