import numpy as np

def dcm2euler(dcm):
    # calculate 321 Euler angles [rad] from DCM
    sin_theta = - dcm[0,2]
    if sin_theta == 1 or sin_theta == -1:
        theta = np.arcsin(sin_theta)
        psi = 0
        sin_phi = -dcm(2,1)
        phi = np.arcsin(sin_phi)
    else:
        theta = np.arcsin(sin_theta)
        phi = np.arctan2(dcm[1,2], dcm[2,2])
        psi = np.arctan2(dcm[0,1], dcm[0,0])

    euler = np.array([psi, theta, phi])
    return euler

# !参考：https://space-denpa.jp/2024/09/23/sinc-best-approx/
# チェビシフ多項式による高次のsinc近似関数
def _chebyshev_sinc(x):
    coefficients = [1, 0, -1/6, 0, 1/120, 0, -1/5040, 0, 1/362880, 0, -1/39916800]
    result = np.zeros_like(x)
    for i, c in enumerate(coefficients):
        result += c * x**i
    return result

# !参考：https://space-denpa.jp/2023/05/31/conversion-quaternion-rotvec/
# チェビシフ近似関数を用いたsinc関数を用いた変換
def to_Lie(quaternion):
    """四元数から回転ベクトルへの変換"""
    theta = 0
    V = quaternion[0:3]
    if quaternion[0] < 0:
        theta = np.arcsin(np.linalg.norm(V))
        rotvec = -2.0/_chebyshev_sinc(theta) * V
    else :
        theta = +np.arcsin(np.linalg.norm(V))
        rotvec = +2.0/_chebyshev_sinc(theta) * V
    return rotvec

def dcm2rotvec(R):
    """
    回転行列 R (3x3) から回転ベクトル（角度 * 回転軸）を計算
    """

    # 回転角 θ を計算
    trace = np.trace(R)
    theta = np.arccos(np.clip((trace - 1) / 2, -1.0, 1.0))  # 数値安定性のためclip

    if np.isclose(theta, 0):
        # 無回転（角度ゼロ）
        return np.zeros(3)

    # 回転軸ベクトルの計算（ロドリゲスの公式より）
    rx = R[2,1] - R[1,2]
    ry = R[0,2] - R[2,0]
    rz = R[1,0] - R[0,1]
    axis = np.array([rx, ry, rz])
    axis = axis / (2 * np.sin(theta))

    # 回転ベクトル = 角度 × 回転軸
    rotvec = theta * axis
    return rotvec

def dcm2quaternion(dcm):
    # calculate quaternion from DCM
    r = dcm2rotvec(dcm) / 2
    r_norm = np.linalg.norm(r)
    V = _chebyshev_sinc(r_norm) * r
    return np.array([V[0], V[1], V[2], np.cos(r_norm)])

def euler2dcm(euler):
    phi   = euler[0] # Z axis Yaw
    theta = euler[1] # Y axis Pitch
    psi   = euler[2] # X axis Roll
    rotx = np.array([[1, 0, 0],
                    [0, np.cos(psi), np.sin(psi)],
                    [0, -np.sin(psi), np.cos(psi)]])
    roty = np.array([[np.cos(theta), 0, -np.sin(theta)],
                    [0, 1, 0],
                    [np.sin(theta), 0, np.cos(theta)]])
    rotz = np.array([[np.cos(phi), np.sin(phi), 0],
                    [-np.sin(phi), np.cos(phi), 0],
                    [0, 0, 1]])
    dcm = rotx @ roty @ rotz
    return dcm

# calculate DCM from quaternion
def quaternion2dcm(q):
    dcm = np.zeros((3,3), dtype=float)
    dcm[0,0] = q[0]*q[0] - q[1]*q[1] - q[2]*q[2] + q[3]*q[3]
    dcm[0,1] = 2 * (q[0]*q[1] + q[2]*q[3])
    dcm[0,2] = 2 * (q[0]*q[2] - q[1]*q[3])
    dcm[1,0] = 2 * (q[0]*q[1] - q[2]*q[3])
    dcm[1,1] = - q[0]*q[0] + q[1]*q[1] - q[2]*q[2] + q[3]*q[3]
    dcm[1,2] = 2 * (q[1]*q[2] + q[0]*q[3])
    dcm[2,0] = 2 * (q[0]*q[2] + q[1]*q[3])
    dcm[2,1] = 2 * (q[1]*q[2] - q[0]*q[3])
    dcm[2,2] = - q[0]*q[0] - q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
    return dcm

# 共役Quaternion計算
def conj_quat(quaternion):
    return np.array([quaternion[0],quaternion[1],quaternion[2],-quaternion[3]])

def Quaternion_product(q, p):
    ans = np.array([[ q[3],-q[2], q[1], q[0]],
                    [ q[2], q[3],-q[0], q[1]],
                    [-q[1], q[0], q[3], q[2]],
                    [-q[0],-q[1],-q[2], q[3]]])
    ans = ans @ p
    return ans
