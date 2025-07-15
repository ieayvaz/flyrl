import numpy as np
from math import sqrt

def calculate_speed(body_vel, attitude):
    V_body = np.array([self.enemy_sim[prp.u_fps], self.enemy_sim[prp.v_fps], self.enemy_sim[prp.w_fps]])
    R_b2n = geoutils.body_to_ned_rotation(self.enemy_sim[prp.roll_rad], self.enemy_sim[prp.pitch_rad], self.enemy_sim[prp.heading_deg]* math.pi / 180)
    V_ned = R_b2n @ V_body
    enemy_speed = np.linalg.norm(V_ned)

def lla_2_enu(geodetic, origin):
    ecef = lla_2_ecef(geodetic)
    ecef_origin = lla_2_ecef(origin)

    return ecef_2_enu(origin, ecef,ecef_origin)

def ecef_2_enu(geodetic, ecef, ecef_origin):
    lat, lon, h = geodetic
    cosLat,cosLon = np.cos(np.radians([lat,lon]))
    sinLat,sinLon = np.sin(np.radians([lat,lon]))
    R = np.array([[-sinLon, cosLon, 0],[-sinLat*cosLon,-sinLat*sinLon, cosLat], [cosLat*cosLon, cosLat*sinLon, sinLat]])
    
    return np.transpose(np.matmul(R,np.atleast_2d(ecef - ecef_origin).T))[0]

def lla_2_ecef(geodetic):
    lat, lon, h = geodetic
    cosLat,cosLon = np.cos(np.radians([lat,lon]))    
    sinLat,sinLon = np.sin(np.radians([lat,lon]))
    a = 6378137.0
    b = 6356752.0
    N = pow(a,2) / sqrt(pow(a,2)*pow(cosLat,2) + pow(b,2)*pow(sinLat,2))
    X = (N + h)*cosLat*cosLon
    Y = (N + h)*cosLat*sinLon
    Z = ((pow(b,2)/pow(a,2))*N + h)*sinLat
    return np.array([X,Y,Z])

def body_to_ned_rotation(phi, theta, psi):
    cφ, sφ = np.cos(phi), np.sin(phi)
    cθ, sθ = np.cos(theta), np.sin(theta)
    cψ, sψ = np.cos(psi), np.sin(psi)

    R = np.array([
        [cθ*cψ, sφ*sθ*cψ - cφ*sψ, cφ*sθ*cψ + sφ*sψ],
        [cθ*sψ, sφ*sθ*sψ + cφ*cψ, cφ*sθ*sψ - sφ*cψ],
        [  -sθ,           sφ*cθ,           cφ*cθ]
    ])
    return R