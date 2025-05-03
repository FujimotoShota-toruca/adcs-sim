# ==== ディレクトリ構成に従って修正した初期コードスケルトン ====

# ==== adcs_library/orbit/propagate.py ====
def propagate_orbit(initial_state, dt, model="two_body"):
    pass

# ==== adcs_library/orbit/tle_parser.py ====
def parse_tle(tle_file_path):
    pass

# ==== adcs_library/attitude/transforms.py ====
def euler_to_quaternion(euler):
    pass

def quaternion_to_dcm(q):
    pass

# ==== adcs_library/attitude/dynamics.py ====
def propagate_attitude(q, w, torque, dt):
    pass

# ==== adcs_library/environment/magnetic_field.py ====
def calculate_geomagnetic_field(position_eci):
    pass

# ==== adcs_library/environment/air_drag.py ====
def calculate_air_drag(position, velocity):
    pass

# ==== adcs_library/environment/gravity.py ====
def calculate_gravity_gradient_torque(position, inertia_tensor, quaternion):
    pass

# ==== adcs_library/control/bdot.py ====
def bdot_control(mag_field, mag_rate):
    pass

# ==== adcs_library/control/quaternion_feedback.py ====
def feedback_control(quaternion, angular_velocity):
    pass

# ==== adcs_library/interfaces/file_io.py → 削除/統合 ====
# 移行先: apps/interface.py

# ==== apps/interface.py ====
def save_output_log(file_path, data):
    pass

def notify_completion():
    pass

# ==== adcs_library/utility/logger.py ====
def setup_logger(output_dir):
    pass

# ==== adcs_library/utility/progress.py ====
def show_progress(current, total):
    pass

# ==== adcs_library/utility/line_notify.py ====
def send_line_notify(message):
    pass

# ==== adcs_library/datatypes/input_types.py ====
from dataclasses import dataclass

@dataclass
class InitialConditions:
    position: list
    velocity: list
    quaternion: list
    angular_velocity: list

# ==== adcs_library/datatypes/output_types.py ====
from dataclasses import dataclass

@dataclass
class SimulationOutput:
    position: list
    attitude: list
    magnetic_field: list

# ==== adcs_library/datatypes/config_model.py ====
from pydantic import BaseModel

class SimulationConfig(BaseModel):
    duration: float
    timestep: float
    save_path: str

# ==== apps/main_simulation.py ====
from adcs_library.datatypes.input_types import InitialConditions
from adcs_library.orbit.propagate import propagate_orbit
from adcs_library.attitude.dynamics import propagate_attitude
from adcs_library.environment.magnetic_field import calculate_geomagnetic_field
from adcs_library.control.bdot import bdot_control
from apps.interface import save_output_log
from adcs_library.utility.logger import setup_logger


def main():
    # 初期化（仮）
    init = InitialConditions(
        position=[7000e3, 0, 0],
        velocity=[0, 7.5e3, 0],
        quaternion=[1, 0, 0, 0],
        angular_velocity=[0, 0, 0]
    )

    dt = 1.0
    steps = 1000
    state = init
    log_data = []

    for _ in range(steps):
        state.position, state.velocity = propagate_orbit(state, dt)
        mag_field = calculate_geomagnetic_field(state.position)
        torque = bdot_control(mag_field, [0, 0, 0])
        state.quaternion, state.angular_velocity = propagate_attitude(
            state.quaternion, state.angular_velocity, torque, dt
        )
        log_data.append(state)

    save_output_log("../output/outputlog.csv", log_data)


if __name__ == "__main__":
    main()
