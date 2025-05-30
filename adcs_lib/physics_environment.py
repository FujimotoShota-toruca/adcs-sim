import numpy as np
import datetime
import ppigrf
import pymap3d as pm
from nrlmsise00 import msise_model
from . import datatype_input

# eci座標系における地球磁場成分
def earth_magnetic_field_eci(_datetime, lon_deg, lat_deg, Alt_km):
    ppigrf_datetime = datetime.datetime(_datetime.year, _datetime.month, _datetime.day)
    B_enu_x, B_enu_y, B_enu_z = ppigrf.igrf(lon_deg, lat_deg, Alt_km, ppigrf_datetime)
    B_enu = np.array([B_enu_x[0], B_enu_y[0], B_enu_z[0]])
    B_norm = np.linalg.norm(np.array(B_enu))
    B_enu = B_enu / B_norm # ノルム保持のための布石
    # emuからecef局所座標に変換
    B_ecef_x, B_ecef_y, B_ecef_z = pm.enu2uvw(B_enu[0], B_enu[1], B_enu[2],lat_deg, lon_deg, deg = True)
    B_ecef = np.array([ B_ecef_x, B_ecef_y, B_ecef_z])
    #print(B_ecef)
    # 座標変換2(ecef2rci)
    B_eci_x, B_eci_y, B_eci_z = pm.ecef2eci(B_ecef[0], B_ecef[1], B_ecef[2], _datetime)
    B_eci = np.array([B_eci_x[0], B_eci_y[0], B_eci_z[0]])
    # ノルム保持
    B_norm = 1.0e-9 * B_norm
    B_enu = B_enu * B_norm 
    B_ecef = B_ecef * B_norm
    B_eci = B_eci * B_norm
    B_ECI = B_eci
    eci_magnetic_field = B_ECI
    return B_enu, B_ecef, B_eci

# 大気抵抗トルクの計算
def atmospheric_torque_body(_datetime: datetime, lon_deg: float, lat_deg: float, Alt_km: float, _f107a: float, _f107: float, _ap: float, body_vel: np.ndarray[np.float64], sat_config: datatype_input.SatelliteStructure):
    V = np.linalg.norm(body_vel)
    body_vel = body_vel/ V
    air_density = msise_model(_datetime, Alt_km, lat_deg, lon_deg, _f107a, _f107, _ap)
    density = air_density[0][5] * 1e3 # total mass density [g cm⁻³]
    area_array = sat_config.area_of_each_surface
    area_position = sat_config.position_vector_of_each_surface
    cg_cp = sat_config.cg_cp_distance
    Cd = sat_config.aerodynamic_drag_coefficient
    air_torque_body = np.array([0.0, 0.0, 0.0])
    projected_area_ratio = [] # 速度に対する面積投影率
    for i, vector in enumerate(sat_config.normal_position_vector_of_each_surface):
        projected_area_ratio.append(np.dot(vector, -body_vel))
        if projected_area_ratio[i] < 0.0:
            projected_area_ratio[i] = 0.0
        air_torque_body += np.cross(area_position[i]-cg_cp, -0.5 * Cd[i] * density * V**2.0 * projected_area_ratio[i] * body_vel * area_array[i])
    
    return air_torque_body

# 重力傾斜トルクの計算
def gravity_gradient_torque_body(_eci_position: np.ndarray[np.float64], _dcm_eci2body: np.ndarray[np.float64], _body_inertia: np.ndarray[np.float64]):
    myu = 3.986e14
    a = 3.0 * myu / (np.linalg.norm(_eci_position)**5)
    u = _dcm_eci2body @ _eci_position # 上の行で先に正規化を項書しているので正規化しなくてよい
    return a * np.cross(u, _body_inertia @ u)