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

def dcm2quaternion(dcm):
    # calculate quaternion from DCM
    q = np.zeros(4, dtype=float)
    q[3] = 0.5 * np.sqrt(1 + dcm[0,0] + dcm[1,1] + dcm[2,2])
    q[0] = 0.25 * (dcm[2,1] - dcm[1,2]) / q[3]
    q[1] = 0.25 * (dcm[0,2] - dcm[2,0]) / q[3]
    q[2] = 0.25 * (dcm[1,0] - dcm[0,1]) / q[3]
    return q

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
