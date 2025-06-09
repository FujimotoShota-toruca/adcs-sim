import numpy as np
from . import math_rotation

# -------------------アクチュエータ挙動模擬-------------------
def discretize_and_limit_moment(MTQ_moment, MTQ_MAX, div_num):

    # 制約条件の適用
    MTQ_moment[0] = min(max(MTQ_moment[0], -MTQ_MAX), MTQ_MAX)
    MTQ_moment[1] = min(max(MTQ_moment[1], -MTQ_MAX), MTQ_MAX)
    MTQ_moment[2] = min(max(MTQ_moment[2], -MTQ_MAX), MTQ_MAX)

    # PWM出力に変更
    MTQ_OUT_x = MTQ_moment[0] / MTQ_MAX
    MTQ_OUT_y = MTQ_moment[1] / MTQ_MAX
    MTQ_OUT_z = MTQ_moment[2] / MTQ_MAX

    MTQ_OUT_x *= div_num
    MTQ_OUT_y *= div_num
    MTQ_OUT_z *= div_num

    MTQ_OUT_x = int(MTQ_OUT_x) / div_num
    MTQ_OUT_y = int(MTQ_OUT_y) / div_num
    MTQ_OUT_z = int(MTQ_OUT_z) / div_num

    MTQ_moment[0] = MTQ_OUT_x * MTQ_MAX
    MTQ_moment[1] = MTQ_OUT_y * MTQ_MAX
    MTQ_moment[2] = MTQ_OUT_z * MTQ_MAX

    return MTQ_moment

def quaternion_to_diff(q_diff):
    # Quaternionを回転行列に変換
    rotation_matrix = math_rotation.quaternion2dcm(q_diff)

    # 基準のX、Y、Z軸方向
    reference_axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # 機体のX、Y、Z軸方向を回転行列で変換
    rotated_axes = np.dot(rotation_matrix, reference_axes.T).T

    # 各軸方向のなす角を計算
    angles = np.arccos(np.sum(reference_axes * rotated_axes, axis=1))

    # ラジアンから度に変換
    angles_deg = np.degrees(angles)

    return angles_deg

def bang_bang_bdot(_x, _MAX_MM):
    sign = 0
    if np.abs(_x) > 1e-7 :
        if _x > 0:
            sign = -1
        else:
            sign = +1
    return sign* _MAX_MM

def generate_axis_vector_LVLH(_eci_position: np.ndarray[np.float64], _eci_velocity: np.ndarray[np.float64]):
    z_axis = -_eci_position / np.linalg.norm(_eci_position)
    y_axis = np.cross(z_axis, _eci_velocity)
    y_axis = y_axis / np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    return x_axis, y_axis, z_axis

def lowpass_filter(filtered_t: np.ndarray, t: np.ndarray, alpha: float) -> np.ndarray:
    """
    単純な指数移動平均（ローパスフィルタ）を適用する関数。

    Parameters:
        filtered_t (np.ndarray): 前回のフィルタ適用後のトルクベクトル（3次元）
        t (np.ndarray): 現在の生のトルクベクトル（3次元）
        alpha (float): フィルタ係数（0 < alpha < 1）。小さいほど平滑化が強い。

    Returns:
        np.ndarray: 新しくフィルタ適用されたトルクベクトル
    """
    return alpha * t + (1 - alpha) * filtered_t
