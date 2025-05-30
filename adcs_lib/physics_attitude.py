import numpy as np

# -------------------姿勢計算ここから-------------------
def skew(vec):
    # create a skew symmetric matrix from a vector
    mat = np.array([[0, -vec[2], vec[1]],
                    [vec[2], 0, -vec[0]],
                    [-vec[1], vec[0], 0]])
    return mat

# convert an inertia around CoG
def inertia_conversion(mass, inertia, cog):
    mat = np.array([[cog[0]*cog[0], cog[0]*cog[1], cog[0]*cog[2]],
                    [cog[1]*cog[0], cog[1]*cog[1], cog[1]*cog[2]],
                    [cog[2]*cog[0], cog[2]*cog[1], cog[2]*cog[2]]])
    inertia_cog = inertia - mass * (np.dot(cog,cog)*np.identity(3) - mat)
    return inertia_cog

# differntial calculation of quaternion
def quaternion_differential(omega, quaternion):
    mat = np.array([[       0,  omega[2], -omega[1],  omega[0]],
                    [-omega[2],        0,  omega[0],  omega[1]],
                    [ omega[1], -omega[0],        0,  omega[2]],
                    [-omega[0], -omega[1], -omega[2],       0]])
    ddt_quaternion = 0.5 * mat @ quaternion
    return ddt_quaternion

# differential calculation of omega
def omega_differential(tau, omega, inertia_cog, inertia_inv):
    ddt_omega = inertia_inv @ (tau - np.cross(omega, inertia_cog @ omega))
    return ddt_omega

# ルンゲ・クッタ法により伝搬
def runge_kutta_quaternion(tau, omega_b, quaternion, inertia_cog, inertia_inv, dt):
    k1 = omega_differential(tau, omega_b, inertia_cog, inertia_inv)
    k2 = omega_differential(tau, omega_b + 0.5*dt*k1, inertia_cog, inertia_inv)
    k3 = omega_differential(tau, omega_b + 0.5*dt*k2, inertia_cog, inertia_inv)
    k4 = omega_differential(tau, omega_b + dt*k3, inertia_cog, inertia_inv)

    l1 = quaternion_differential(omega_b, quaternion)
    l2 = quaternion_differential(omega_b + 0.5*dt*k1, quaternion + 0.5*dt*l1)
    l3 = quaternion_differential(omega_b + 0.5*dt*k2, quaternion + 0.5*dt*l2)
    l4 = quaternion_differential(omega_b + dt*k3, quaternion + dt*l3)

    new_omega_b = omega_b + 1/6 * (k1 + 2*k2 + 2*k3 + k4) * dt
    new_quaternion = quaternion + 1/6 * (l1 + 2*l2 + 2*l3 + l4) * dt
    new_quaternion = new_quaternion / np.linalg.norm(new_quaternion)

    return new_omega_b, new_quaternion

def ned_to_ecef_matrix(lat, lon):
    lat = np.radians(lat)
    lon = np.radians(lon)

    R = np.array([
        [-np.sin(lon), np.cos(lon), 0],
        [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)],
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    ])
    return R