from math import cos, exp, isnan, pi, sin
import math
import random
from typing import Dict, Sequence, Tuple
import numpy as np
from flyrl.aircraft import Aircraft
from flyrl.ap_simulation import AP_Simulation
from flyrl.ap_tasks import BaseAPTask
from flyrl.basic_tasks import BaseFlightTask
from flyrl.properties import BoundedProperty, DerivedProperty, Property
from flyrl.simulation import Simulation
import flyrl.properties as prp
from flyrl import geoutils
from flyrl.target import Target
from flyrl.utils import angle_between, mt2ft, ft2mt
from flyrl.autopilot import AutoPilot

import matplotlib.pyplot as plt

def get_enu_to_body_rotation_matrix(roll_rad, pitch_rad, heading_rad):
    # ... (keep the existing function implementation) ...
    phi = roll_rad
    theta = pitch_rad
    psi = heading_rad # Assuming flyrl heading is standard yaw for rotation purposes here. Might need check.

    cos_phi = cos(phi)
    sin_phi = sin(phi)
    cos_theta = cos(theta)
    sin_theta = sin(theta)
    cos_psi = cos(psi)
    sin_psi = sin(psi)

    # Rotation: Body = R_bi * ENU
    # Using standard ZYX sequence transpose:
    R_body_enu = np.array([
        [cos_theta * cos_psi, cos_theta * sin_psi, -sin_theta],
        [sin_phi * sin_theta * cos_psi - cos_phi * sin_psi, sin_phi * sin_theta * sin_psi + cos_phi * cos_psi, sin_phi * cos_theta],
        [cos_phi * sin_theta * cos_psi + sin_phi * sin_psi, cos_phi * sin_theta * sin_psi - sin_phi * cos_psi, cos_phi * cos_theta]
    ])
    return R_body_enu


class DogfightTask(BaseFlightTask):
    INITIAL_HEADING_DEG = 120.0
    THROTTLE_CMD = 0.8 # Default throttle level. TODO: Make this an action
    MIXTURE_CMD = 0.8

    GROUND_LEVEL = 520
    # --- Normalization constants (adjust based on aircraft and scenario) ---
    MAX_ENGAGEMENT_RANGE_M = 2000.0
    MAX_RELATIVE_SPEED_MPS = 500.0 # Max relative speed expected
    MAX_CLOSURE_RATE_MPS = 300.0
    # Removed Airspeed constants
    MAX_ALPHA_DEG = 15.0
    MIN_ALPHA_DEG = -5.0
    MAX_BETA_DEG = 10.0
    # Removed G constants
    MAX_ROLL_RATE_RPS = pi # 180 deg/s
    MAX_PITCH_RATE_RPS = pi/2 # 90 deg/s
    MAX_YAW_RATE_RPS = pi/3 # 60 deg/s

    # --- Reward function constants (CRITICAL TUNING PARAMETERS) ---
    W_GEOM = 1.5          # Weight for geometric advantage (AOB * AA)
    K_AOB = 3.0           # Steepness for AOB reward curve
    K_AA = 2.0            # Steepness for AA reward curve
    W_RANGE = 0.5         # Weight for range control reward
    OPTIMAL_RANGE_M = 300.0 # Desired engagement distance
    K_RANGE = 5e-5        # Width of range reward gaussian (smaller k = wider)
    W_CLOSURE = 0.3       # Weight for closure rate reward
    K_CLOSURE_FAR = 0.1   # Factor for desired closure when far
    K_CLOSURE_PENALTY = 1e-4 # Penalty for deviating from desired closure
    W_ALIVE = 0.01        # Small reward per step for staying alive
    W_AOA = 0.1           # Penalty weight for high AoA
    AOA_THRESHOLD_DEG = 15.0 # AoA limit before penalty starts
    # Removed G penalty weight and threshold
    W_BETA = 0.05         # Penalty weight for sideslip
    R_LOCK_SUCCESS = 100.0 # Terminal reward for achieving lock
    R_CRASH = -100.0       # Terminal penalty for crashing
    R_TIMEOUT = -10.0      # Terminal penalty for timeout
    R_OUT_OF_BOUNDS = -50.0# Terminal penalty for leaving area (implement OOB check)

    # --- Lock condition parameters ---
    LOCK_AOB_DEG = 10.0
    LOCK_AA_DEG = 20.0
    LOCK_RANGE_M = 500.0
    # LOCK_SUSTAIN_STEPS = 5 # TODO: Implement sustained lock check if needed


    def __init__(self, step_frequency_hz: float, sim: Simulation, aircraft: Aircraft, max_time_s: float = 120.0, debug: bool = False):
        self.target = Target("~/flyrl/flight_data2.csv") # Assuming this provides Lat, Lon, Alt, Roll, Pitch, Heading

        # --- Define State Variables (Removed airspeed and Nz) ---
        # Relative Position (Body Frame) - Normalized [-1, 1]
        rel_pos_forward = DerivedProperty("rel_pos_forward", "Relative pos forward (norm)", -1, 1)
        rel_pos_right = DerivedProperty("rel_pos_right", "Relative pos right (norm)", -1, 1)
        rel_pos_down = DerivedProperty("rel_pos_down", "Relative pos down (norm)", -1, 1)
        # Relative Velocity (ENU Frame) - Normalized [-1, 1]
        rel_vel_x_enu = DerivedProperty("rel_vel_x_enu", "Relative vel East (norm)", -1, 1)
        rel_vel_y_enu = DerivedProperty("rel_vel_y_enu", "Relative vel North (norm)", -1, 1)
        rel_vel_z_enu = DerivedProperty("rel_vel_z_enu", "Relative vel Up (norm)", -1, 1)
        # Key Angles (Radians) - Normalized [0, 1]
        angle_off_boresight = DerivedProperty("AOB", "Angle off Boresight (norm)", 0, 1) # Range [0, pi] -> [0, 1]
        aspect_angle = DerivedProperty("AA", "Aspect Angle (norm)", 0, 1) # Range [0, pi] -> [0, 1]
        # Closure Rate - Normalized [-1, 1]
        closure_rate = DerivedProperty("closure_rate_mps_norm", "Closure rate (norm)", -1, 1)
        # Own Aircraft State - Normalized [-1, 1] or [0, 1]
        # REMOVED: own_airspeed_norm
        own_alpha_norm = DerivedProperty("own_alpha_norm", "Own AoA (norm)", -1, 1)
        own_beta_norm = DerivedProperty("own_beta_norm", "Own Beta (norm)", -1, 1)
        # REMOVED: own_Nz_norm
        own_roll_rate_norm = DerivedProperty("own_p_norm", "Own roll rate (norm)", -1, 1)
        own_pitch_rate_norm = DerivedProperty("own_q_norm", "Own pitch rate (norm)", -1, 1)
        own_yaw_rate_norm = DerivedProperty("own_r_norm", "Own yaw rate (norm)", -1, 1)
        own_sin_roll = DerivedProperty("own_sin_roll", "sin(roll)", -1, 1)
        own_cos_roll = DerivedProperty("own_cos_roll", "cos(roll)", -1, 1)
        own_pitch_norm = DerivedProperty("own_pitch_norm", "Own pitch (norm)", -1, 1)


        # Update the state_variables tuple
        super().__init__(state_variables=(rel_pos_forward, rel_pos_right, rel_pos_down,
                                         rel_vel_x_enu, rel_vel_y_enu, rel_vel_z_enu,
                                         angle_off_boresight, aspect_angle, closure_rate,
                                         # own_airspeed_norm removed
                                         own_alpha_norm, own_beta_norm,
                                         # own_Nz_norm removed
                                         own_roll_rate_norm, own_pitch_rate_norm,
                                         own_yaw_rate_norm, own_sin_roll, own_cos_roll,
                                         own_pitch_norm),
                         max_time_s=max_time_s,
                         step_frequency_hz=step_frequency_hz,
                         use_autopilot=True,
                         sim=sim, debug=debug)
        self.aircraft = aircraft

        # Cache for kinematic calculations to avoid redundant computation per step
        self._kinematics_cache = {}
        self._cache_step = -1 # Track sim step for cache validity

    def _update_kinematics_cache(self):
        # ... (Keep the existing implementation - it doesn't use airspeed/Nz directly) ...
        current_sim_step = self.sim.get_sim_time() * self.sim.sim_frequency_hz # Approximate step count
        if self._cache_step == current_sim_step:
            return # Already calculated for this step

        # --- 1. Get Own State ---
        own_lat = self.sim[prp.lat_geod_deg]
        own_lon = self.sim[prp.lng_geoc_deg]
        own_alt = self.get_prop(prp.altitude_sl_mt)
        own_geo_pos = np.array([own_lat, own_lon, own_alt])
        own_enu_pos = geoutils.lla_2_enu(own_geo_pos, self.origin)

        own_v_east_mps = self.sim[prp.v_east_fps] * ft2mt(1.0)
        own_v_north_mps = self.sim[prp.v_north_fps] * ft2mt(1.0)
        own_v_up_mps = -self.sim[prp.v_down_fps] * ft2mt(1.0) # Convert v_down to v_up
        own_enu_vel = np.array([own_v_east_mps, own_v_north_mps, own_v_up_mps])

        own_roll = self.sim[prp.roll_rad]
        own_pitch = self.sim[prp.pitch_rad]
        own_heading = self.sim[prp.heading_deg]*pi/180.0 # Check if this is rad

        # --- 2. Get Target State ---
        target_lat = self.target["Lat"]
        target_lon = self.target["Lon"]
        target_alt = self.target["Alt"]
        target_geo_pos = np.array([target_lat, target_lon, target_alt])
        target_enu_pos = geoutils.lla_2_enu(target_geo_pos, self.origin)

        if hasattr(self.target, 'vx_enu') and hasattr(self.target, 'vy_enu') and hasattr(self.target, 'vz_enu'):
             target_enu_vel = np.array([self.target.vx_enu, self.target.vy_enu, self.target.vz_enu])
        else:
             if hasattr(self.target, '_last_enu_pos') and self.target._last_update_time > 0:
                 dt = 1.0 / self.step_frequency_hz # Approx time since last update
                 target_enu_vel = (target_enu_pos - self.target._last_enu_pos) / dt if dt > 0 else np.zeros(3)
             else:
                 target_enu_vel = np.zeros(3) # Default to zero velocity if no history
             self.target._last_enu_pos = target_enu_pos.copy()
             self.target._last_update_time = self.sim.get_sim_time()

        target_roll = self.target["Roll"]
        target_pitch = self.target["Pitch"]
        target_heading = self.target["Heading"]

        # --- 3. Calculate Relative Kinematics ---
        rel_enu_pos = target_enu_pos - own_enu_pos
        rel_enu_vel = target_enu_vel - own_enu_vel
        distance = np.linalg.norm(rel_enu_pos)
        distance = max(distance, 1e-6)

        # --- 4. Calculate Angles ---
        vec_to_target_enu = rel_enu_pos
        R_enu_body = get_enu_to_body_rotation_matrix(own_roll, own_pitch, own_heading).T
        own_nose_vector_body = np.array([1, 0, 0])
        own_nose_vector_enu = R_enu_body @ own_nose_vector_body
        aob_rad = angle_between(own_nose_vector_enu, vec_to_target_enu)

        R_target_enu_body = get_enu_to_body_rotation_matrix(target_roll, target_pitch, target_heading).T
        target_tail_vector_body = np.array([-1, 0, 0])
        target_tail_vector_enu = R_target_enu_body @ target_tail_vector_body
        vec_to_ownship_enu = -rel_enu_pos
        aa_rad = angle_between(target_tail_vector_enu, vec_to_ownship_enu)

        # --- 5. Calculate Closure Rate ---
        closure_rate_mps = -np.dot(rel_enu_pos, rel_enu_vel) / distance

        # --- 6. Calculate Relative Position in Body Frame ---
        R_body_enu = R_enu_body.T
        rel_body_pos = R_body_enu @ rel_enu_pos
        rel_pos_forward_val = rel_body_pos[0]
        rel_pos_right_val = rel_body_pos[1]
        rel_pos_down_val = rel_body_pos[2]

        # --- 7. Store in Cache ---
        self._kinematics_cache = {
            'rel_pos_forward': rel_pos_forward_val,
            'rel_pos_right': rel_pos_right_val,
            'rel_pos_down': rel_pos_down_val,
            'rel_vel_x_enu': rel_enu_vel[0],
            'rel_vel_y_enu': rel_enu_vel[1],
            'rel_vel_z_enu': rel_enu_vel[2],
            'AOB': aob_rad,
            'AA': aa_rad,
            'closure_rate_mps': closure_rate_mps,
            'distance': distance,
            'own_roll': own_roll,
            'own_pitch': own_pitch,
        }
        self._cache_step = current_sim_step


    # --- Normalization Helper --- Keep as is
    def _normalize(self, value, min_val, max_val, clip=True):
        # ... (keep implementation) ...
        if max_val == min_val: return 0.0
        norm_val = 2 * (value - min_val) / (max_val - min_val) - 1
        if clip: return np.clip(norm_val, -1.0, 1.0)
        return norm_val

    def _normalize_positive(self, value, max_val, clip=True):
        # ... (keep implementation) ...
        if max_val <= 0: return 0.0
        norm_val = value / max_val
        if clip: return np.clip(norm_val, 0.0, 1.0)
        return norm_val


    def get_prop(self, prop) -> float:
        # Try base class properties first
        value = super().get_prop(prop)

        # Handle raw values needed for later normalization (AoA, Beta, Rates)
        if value is not None:
            return value

        # If super().get_prop(prop) returned None, process derived properties

        # Ensure kinematic cache is up-to-date
        self._update_kinematics_cache()
        cache = self._kinematics_cache

        value = None # Initialize value for this block

        # --- Process Derived State Properties (Removed airspeed and Nz) ---
        if prop.name == "rel_pos_forward":
            value = cache.get('rel_pos_forward', 0.0) / self.MAX_ENGAGEMENT_RANGE_M
            value = np.clip(value, -1.0, 1.0)
        elif prop.name == "rel_pos_right":
            value = cache.get('rel_pos_right', 0.0) / self.MAX_ENGAGEMENT_RANGE_M
            value = np.clip(value, -1.0, 1.0)
        elif prop.name == "rel_pos_down":
            value = cache.get('rel_pos_down', 0.0) / self.MAX_ENGAGEMENT_RANGE_M
            value = np.clip(value, -1.0, 1.0)
        elif prop.name == "rel_vel_x_enu":
            raw_value = cache.get('rel_vel_x_enu', 0.0)
            value = self._normalize(raw_value, -self.MAX_RELATIVE_SPEED_MPS, self.MAX_RELATIVE_SPEED_MPS)
        elif prop.name == "rel_vel_y_enu":
            raw_value = cache.get('rel_vel_y_enu', 0.0)
            value = self._normalize(raw_value, -self.MAX_RELATIVE_SPEED_MPS, self.MAX_RELATIVE_SPEED_MPS)
        elif prop.name == "rel_vel_z_enu":
            raw_value = cache.get('rel_vel_z_enu', 0.0)
            value = self._normalize(raw_value, -self.MAX_RELATIVE_SPEED_MPS, self.MAX_RELATIVE_SPEED_MPS)
        elif prop.name == "AOB":
            value = cache.get('AOB', pi) / pi
            value = np.clip(value, 0.0, 1.0)
        elif prop.name == "AA":
            value = cache.get('AA', pi) / pi
            value = np.clip(value, 0.0, 1.0)
        elif prop.name == "closure_rate_mps_norm":
            raw_value = cache.get('closure_rate_mps', 0.0)
            value = self._normalize(raw_value, -self.MAX_CLOSURE_RATE_MPS, self.MAX_CLOSURE_RATE_MPS)
        # REMOVED elif for own_airspeed_norm
        elif prop.name == "own_alpha_norm":
            raw_value = self.sim[prp.alpha_deg]
            value = self._normalize(raw_value, self.MIN_ALPHA_DEG, self.MAX_ALPHA_DEG)
        elif prop.name == "own_beta_norm":
            raw_value = self.sim[prp.sideslip_deg]
            value = self._normalize(raw_value, -self.MAX_BETA_DEG, self.MAX_BETA_DEG)
        # REMOVED elif for own_Nz_norm
        elif prop.name == "own_p_norm":
            raw_value = self.sim[prp.p_radps]
            value = self._normalize(raw_value, -self.MAX_ROLL_RATE_RPS, self.MAX_ROLL_RATE_RPS)
        elif prop.name == "own_q_norm":
            raw_value = self.sim[prp.q_radps]
            value = self._normalize(raw_value, -self.MAX_PITCH_RATE_RPS, self.MAX_PITCH_RATE_RPS)
        elif prop.name == "own_r_norm":
            raw_value = self.sim[prp.r_radps]
            value = self._normalize(raw_value, -self.MAX_YAW_RATE_RPS, self.MAX_YAW_RATE_RPS)
        elif prop.name == "own_sin_roll":
             value = sin(cache.get('own_roll', self.sim[prp.roll_rad]))
        elif prop.name == "own_cos_roll":
             value = cos(cache.get('own_roll', self.sim[prp.roll_rad]))
        elif prop.name == "own_pitch_norm":
             raw_value = cache.get('own_pitch', self.sim[prp.pitch_rad])
             value = self._normalize(raw_value, -pi/2, pi/2)

        # Final check and clipping
        if value is not None:
            if hasattr(prop, 'min') and hasattr(prop, 'max') and prop.min is not None and prop.max is not None:
                 value = np.clip(float(value), prop.min, prop.max)
            if isnan(value):
                 print(f"Warning: NaN detected for property {prop.name} after calculation/normalization. Returning 0.")
                 value = 0.0
            return float(value)
        else:
             print(f"Warning: Property {prop.name} requested but not handled in get_prop. Returning 0.0")
             return 0.0


    def task_step(self, action: Sequence[float], sim_steps: int) \
        -> Tuple[np.ndarray, float, bool, Dict]:

        # --- Update Target ---
        self.target.step(1.0 / self.step_frequency_hz)

        state, reward, done,_ = super().task_step(action, sim_steps)

        # --- Check Terminal Conditions ---
        is_locked = self._check_lock_condition(self._kinematics_cache)
        crashed = self._check_crash_condition()
        out_of_bounds = self._check_oob_condition(self._kinematics_cache)
        timeout = self.max_steps - self.current_step <= 0
        sim_error = any(isnan(s) for s in state)

        # --- Debug Output (Removed Nz) ---
        if self.debug and self.current_step % 10 == 0:
            print(f"Step: {self.current_step}, Time: {self.current_step*(1.0/self.step_frequency_hz):.2f}")
            print(f"  Dist: {self._kinematics_cache.get('distance', 0):.1f}m, "
                  f"AOB: {math.degrees(self._kinematics_cache.get('AOB', 0)):.1f}deg, "
                  f"AA: {math.degrees(self._kinematics_cache.get('AA', 0)):.1f}deg")
            # Get True Airspeed (u_fps) for logging if needed, convert to kts
            tas_kts = (self.sim[prp.u_fps] * ft2mt(1.0)) / 0.514444 if prp.u_fps else 0
            print(f"  Own State: Alt={self.get_prop(prp.altitude_sl_mt):.1f}m, "
                  # f"V={self.sim[prp.indicated_airspeed_kt]:.1f}kts, " # Removed KIAS
                  f"TAS={tas_kts:.1f}kts, " # Log True Airspeed instead?
                  f"AoA={self.sim[prp.alpha_deg]:.1f}deg") # Removed Nz
            print(f"  Reward: {reward:.3f}, Done: {done} (Lock={is_locked}, Crash={crashed}, OOB={out_of_bounds}, Timeout={timeout}, Error={sim_error})")


        return np.array(state, dtype=np.float32), reward, done, {}

    def _is_terminal(self, state, sim):
        is_locked = self._check_lock_condition(self._kinematics_cache)
        crashed = self._check_crash_condition()
        out_of_bounds = self._check_oob_condition(self._kinematics_cache)
        timeout = self.max_steps - self.current_step <= 0
        sim_error = any(isnan(s) for s in state)

        done = is_locked or crashed or out_of_bounds or timeout or sim_error
        return super()._is_terminal(state, sim) or done

    def get_initial_conditions(self):
        # ... (Keep the previously corrected implementation) ...
        self.target.reset() 
        target_start_lat = self.target["Lat"]
        target_start_lon = self.target["Lon"]
        target_start_alt_m = self.target["Alt"] 

        initial_heading_deg = np.random.uniform(0, 360) 
        # Use true airspeed (u_fps) from aircraft definition if available
        initial_airspeed_fps = self.aircraft.get_cruise_speed_fps() * np.random.uniform(0.9, 1.1) 

        initial_alt_m = target_start_alt_m + np.random.uniform(-150, 150) 
        initial_alt_ft = mt2ft(initial_alt_m)
        initial_alt_ft = max(initial_alt_ft, self.base_initial_conditions.get(prp.initial_terrain_altitude_ft, 0) + 100) 

        lat_offset_deg = 0.015 
        lon_offset_deg = 0.020 
        initial_lat = target_start_lat + np.random.uniform(-lat_offset_deg, lat_offset_deg)
        initial_lon = target_start_lon + np.random.uniform(-lon_offset_deg, lon_offset_deg)

        conditions = {
            prp.initial_latitude_geod_deg: initial_lat,
            prp.initial_longitude_geoc_deg: initial_lon,
            prp.initial_altitude_ft: initial_alt_ft,
            prp.initial_heading_deg: initial_heading_deg,
            prp.initial_u_fps: initial_airspeed_fps, # Set true airspeed
            prp.initial_v_fps: 0,
            prp.initial_w_fps: 0,
            prp.initial_p_radps: 0,
            prp.initial_q_radps: 0,
            prp.initial_r_radps: 0,
            prp.initial_roc_fpm: 0,
            prp.initial_terrain_altitude_ft: self.base_initial_conditions.get(prp.initial_terrain_altitude_ft, 0.00000001),
        }
        return conditions


    def _new_episode_init(self):
        # ... (Keep the existing implementation) ...
        self._cache_step = -1
        self._kinematics_cache = {}
        super()._new_episode_init() 
        self.origin = np.array([self.sim[prp.lat_geod_deg],
                                self.sim[prp.lng_geoc_deg],
                                self.get_prop(prp.altitude_sl_mt)])
        self.target._last_enu_pos = geoutils.lla_2_enu(np.array([self.target["Lat"], self.target["Lon"], self.target["Alt"]]), self.origin)
        self.target._last_update_time = 0
        self.sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)


    def _check_lock_condition(self, cache) -> bool:
        # ... (Keep the existing implementation) ...
        if not cache: return False
        aob_ok = cache.get('AOB', pi) <= math.radians(self.LOCK_AOB_DEG)
        aa_ok = cache.get('AA', pi) <= math.radians(self.LOCK_AA_DEG)
        range_ok = cache.get('distance', self.MAX_ENGAGEMENT_RANGE_M + 1) <= self.LOCK_RANGE_M
        return aob_ok and aa_ok and range_ok

    def _check_crash_condition(self) -> bool:
        # ... (Keep the existing implementation - doesn't use Nz) ...
        altitude_agl_mt = self.get_prop(prp.altitude_sl_mt) - self.GROUND_LEVEL
        max_pitch = abs(self.sim[prp.pitch_rad]) > math.radians(85)
        max_roll = abs(self.sim[prp.roll_rad]) > math.radians(150) 
        return altitude_agl_mt < 20.0 or max_pitch or max_roll

    def _check_oob_condition(self, cache) -> bool:
        # ... (Keep the existing implementation) ...
        if not cache: return False
        return cache.get('distance', 0) > self.MAX_ENGAGEMENT_RANGE_M * 1.5

    def calculate_reward(self, state_values_passed_in, done, is_locked, crashed, out_of_bounds, timeout, sim_error) -> float:
        """ Calculates the reward based on the designed function (No Nz penalty)."""

        # --- Terminal Rewards --- (Keep as is)
        if done:
            if is_locked: return self.R_LOCK_SUCCESS
            if crashed: return self.R_CRASH
            if out_of_bounds: return self.R_OUT_OF_BOUNDS
            if timeout: return self.R_TIMEOUT
            if sim_error: return self.R_CRASH
            return 0.0

        # --- Shaping Rewards ---
        cache = self._kinematics_cache
        if not cache: return 0.0

        # 1. Geometric Advantage Reward (R_geom) - Keep as is
        aob_rad = cache.get('AOB', pi)
        aa_rad = cache.get('AA', pi)
        r_aob = exp(-self.K_AOB * aob_rad)
        r_aa = exp(-self.K_AA * aa_rad)
        r_geom = self.W_GEOM * r_aob * r_aa

        # 2. Range Control Reward (R_range) - Keep as is
        distance = cache.get('distance', self.MAX_ENGAGEMENT_RANGE_M)
        r_range = self.W_RANGE * exp(-self.K_RANGE * (distance - self.OPTIMAL_RANGE_M)**2)

        # 3. Closure Rate Reward (R_closure) - Keep as is (though MAX_AIRSPEED_MPS removed, might need adjustment for desired_closure cap if that was important)
        closure_rate_mps = cache.get('closure_rate_mps', 0)
        desired_closure = self.K_CLOSURE_FAR * max(0, distance - self.OPTIMAL_RANGE_M)
        # Consider if a cap based on relative speed or a fixed value is needed here now
        # desired_closure = min(desired_closure, some_reasonable_max_closure)
        r_closure = self.W_CLOSURE * exp(-self.K_CLOSURE_PENALTY * (closure_rate_mps - desired_closure)**2)

        # 4. Flight Envelope Protection Penalties (R_penalty) - Removed G penalty
        own_alpha_deg = self.sim[prp.alpha_deg]
        # REMOVED: own_Nz = self.sim[prp.accelerations_Nz]
        own_beta_deg = self.sim[prp.sideslip_deg]

        p_aoa = -self.W_AOA * max(0, abs(own_alpha_deg) - self.AOA_THRESHOLD_DEG)**2
        # REMOVED: p_gload = -self.W_GLOAD * max(0, abs(own_Nz) - self.G_THRESHOLD)**2
        p_beta = -self.W_BETA * abs(own_beta_deg)
        r_penalty = p_aoa + p_beta # Only AoA and Beta penalties remain

        # 5. Alive Bonus / Time Penalty (R_alive) - Keep as is
        r_alive = self.W_ALIVE

        # --- Total Reward ---
        total_reward = r_geom + r_range + r_closure + r_penalty + r_alive

        if isnan(total_reward):
             print(f"Warning: NaN reward calculated. Components: geom={r_geom:.2f}, range={r_range:.2f}, closure={r_closure:.2f}, penalty={r_penalty:.2f}, alive={r_alive:.2f}")
             total_reward = -1.0
        total_reward = np.clip(total_reward, -10.0, 10.0)

        return total_reward

    # --- Helper methods --- (Keep get_geo_pos, get_target_geo, get_pos, get_target_pos)
    def get_geo_pos(self): # Keep
        return np.array([self.sim[prp.lat_geod_deg], self.sim[prp.lng_geoc_deg], self.get_prop(prp.altitude_sl_mt)])
    def get_target_geo(self): # Keep
        return np.array([self.target["Lat"], self.target["Lon"], self.target["Alt"]])
    def get_pos(self): # Keep
        if not hasattr(self, 'origin') or self.origin is None: self._new_episode_init()
        return geoutils.lla_2_enu(self.get_geo_pos(), self.origin)
    def get_target_pos(self): # Keep
        if not hasattr(self, 'origin') or self.origin is None: self._new_episode_init()
        return geoutils.lla_2_enu(self.get_target_geo(), self.origin)

    def get_props_to_output(self):
        # Removed prp.Nz, kept u_fps (True Airspeed component)
        return (prp.u_fps, prp.altitude_sl_ft, prp.lat_geod_deg, prp.lng_geoc_deg,
                prp.heading_deg, prp.roll_rad, prp.pitch_rad, prp.alpha_deg)
    
class DogfightAP2PlayerTask(BaseAPTask):
    INITIAL_HEADING_DEG = 120.0
    THROTTLE_CMD = 0.5 # Default throttle level.
    MIXTURE_CMD = 0.5

    TARGET_ADDRESS = "127.0.0.1:14560"

    def __init__(self, step_frequency_hz: float, sim: Simulation, aircraft: Aircraft, max_time_s: float = 300.0, debug: bool = False):
        distance = DerivedProperty("distance", "Distance between target and airplane",0,2500)
        distance_x = DerivedProperty("distance_x","Distance between target and airplane in x axis",-1000,1000)
        distance_y = DerivedProperty("distance_y","Distance between target and airplane in y axis",-1000,1000)
        distance_z = DerivedProperty("distance_z","Distance between target and airplane in z axis",-250,250)
        enemy_roll = DerivedProperty("enemy_roll_rad","Enemy roll angle in radians",-pi,pi)
        enemy_pitch = DerivedProperty("enemy_pitch_rad","Enemy pitch angle in radians",-pi/2,pi/2)
        enemy_heading = DerivedProperty("enemy_heading_deg","enemy heading angle in degrees", 0, 360)
        los_error = DerivedProperty("los_error", "Angle between LOS and airplane heading in degrees",0,360)
        target_roll = DerivedProperty("target_roll_deg_norm", "Target roll for aiplane in degrees", -1,1)
        target_pitch = DerivedProperty("target_pitch_deg_norm", "Target pitch for aiplane in degrees", -1,1)
        self.target = AP_Simulation(self.TARGET_ADDRESS,controlled=False)
        super().__init__((distance, los_error),
                         max_time_s, step_frequency_hz, sim,debug=debug, action_variables=(target_roll,), autopilot=True)
        self.aircraft = aircraft
        self.los_error_prp = los_error
        self.los_prp = DerivedProperty("los","Los angle",0,360)
        self.enemy_heading_prp = DerivedProperty("enemy_heading_deg","Enemy heading",0,360)
        self.distance_prp = distance

    def task_step(self, action: Sequence[float], sim_steps: int) \
        -> Tuple[np.ndarray, float, bool, Dict]:
        if self.debug:
            out_props = [self.distance_prp,self.los_error_prp, self.los_prp, prp.heading_deg, self.enemy_heading_prp]
            for _prop in out_props:
                print(f"{_prop.name} : {self.get_prop(_prop)}")

            print("\n")
        
        return super().task_step(action, sim_steps)
        
    def _new_episode_init(self):
        super()._new_episode_init()
        self.sim.set_throttle(self.THROTTLE_CMD)
        # Set origin of coordinate system to inital position of airplane
        self.origin = np.array([self.get_prop(_prop) for _prop in [prp.lat_geod_deg,
                                                                   prp.lng_geoc_deg,
                                                                   prp.altitude_sl_mt]])
        #TODO: set target plane?

    def get_initial_conditions(self):
        return super().get_initial_conditions()

    def is_locked(self):
        if self.get_distance() > 50.0:
            return False
        dist_azimuth = math.atan2(self.get_distance_v()[0],self.get_distance_v()[1]) % 360
        if abs(self.sim[prp.heading_deg] - dist_azimuth) > 20.0:
            return False
        return True
    
    def distance_limit(self):
        return self.get_distance() > 10000
    
    def get_los(self):
        dist = self.get_distance_v()
        return math.degrees(math.atan2(dist[0],dist[1])) % 360
    
    def get_los_error(self):
        return ((self.get_los() - self.get_heading()) + 180) % 360 - 180

    def get_heading(self):
        return self.sim[prp.heading_deg]
    
    def altitude_limit(self):
        return self.get_prop(prp.altitude_sl_mt) < 600

    def get_geo_pos(self):
        return np.array([self.get_prop(_prop) for _prop in [prp.lat_geod_deg,
                                                            prp.lng_geoc_deg,
                                                            prp.altitude_sl_mt]])
    def get_target_geo(self):
        return np.array([self.target[prp.lat_geod_deg],self.target[prp.lng_geoc_deg],self.target[prp.altitude_sl_mt]])

    def get_pos(self):
        return geoutils.lla_2_enu(self.get_geo_pos(),self.origin)
    
    def get_target_pos(self):
        return geoutils.lla_2_enu(self.get_target_geo(),self.origin)

    def get_props_to_output(self):
        return (prp.roll_rad, prp.pitch_rad)

    def get_distance(self):
        return np.linalg.norm(self.get_target_pos() - self.get_pos())

    def get_distance_v(self):
        return self.get_target_pos() - self.get_pos() 

    def get_prop(self, prop) -> float:
        value = super().get_prop(prop)
        if value != None:
            return value
        # Process derived properties  
        if prop.name == "distance":
            value = float(self.get_distance())
        if prop.name == "distance_x":
            value = float(self.get_distance_v()[0])
        if prop.name == "distance_y":
            value = float(self.get_distance_v()[1])
        if prop.name == "distance_z":
            value = float(self.get_distance_v()[2])
        '''if prop.name == "target_heading":
            return self.target["Heading"]'''
        if prop.name == "enemy_roll_rad":
            value = float(self.target[prp.roll_rad])
        if prop.name == "enemy_pitch_rad":
            value = float(self.target[prp.pitch_rad])
        if prop.name == "enemy_heading_deg":
            value = float(self.target[prp.heading_deg])
        if prop.name == "los_error":
            value = self.get_los_error()
        if prop.name == "los":
            value = self.get_los()
        return value
        
    def _is_terminal(self,state, sim: Simulation) -> bool:
        term_cond = self.is_locked() or self.distance_limit() or self.altitude_limit()
        return super()._is_terminal(state,sim) or term_cond
    
    def calculate_reward(self, state):
        rew = 0.0
        if self.is_locked():
            return 100
        #rew += 0.01 * (500/(self.get_distance()) - 1)
        rew += (90 - abs(self.get_los_error())) / (90.0*2.5)
        
        return rew
        