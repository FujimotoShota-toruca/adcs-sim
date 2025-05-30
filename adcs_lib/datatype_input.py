from dataclasses import dataclass
from datetime import datetime
import numpy as np

@dataclass
class Time:
    start: datetime
    duration: float  # 秒数 [sec]
    step: float      # 秒数 [sec]

@dataclass
class InitialAttitudeParameters:
    angular_velocity: np.ndarray
    quaternion: np.ndarray

@dataclass
class SatelliteStructure:
    inertia_tensor: np.ndarray
    area_of_each_surface: np.ndarray
    aerodynamic_drag_coefficient: np.ndarray
    position_vector_of_each_surface: np.ndarray
    normal_position_vector_of_each_surface: np.ndarray
    cg_cp_distance: np.ndarray

@dataclass
class Config:
    tle: str
    time: Time
    initial_attitude_parameters: InitialAttitudeParameters
    satellite_structure: SatelliteStructure