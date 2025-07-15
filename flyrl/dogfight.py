from contextlib import contextmanager
from math import cos, exp, isnan, pi, sin
import math
import os
import random
import sys
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

from flyrl.visualizer_dogfight import DogfightVisualizer

class DogfightTask(BaseFlightTask):
    INITIAL_HEADING_DEG = 120.0
    GROUND_ALTITUDE_MT = 575
    THROTTLE_CMD = 0.8 # Default throttle level.
    MIXTURE_CMD = 0.8

    def __init__(self, step_frequency_hz: float, sim: Simulation, aircraft: Aircraft, max_time_s: float = 120.0, debug: bool = False):
        self.target = Target("~/flyrl/flight_data2.csv")
        distance = DerivedProperty("distance", "Distance between target and airplane",0,2500)
        distance_x = DerivedProperty("distance_x","Distance between target and airplane in x axis",-1000,1000)
        distance_y = DerivedProperty("distance_y","Distance between target and airplane in y axis",-1000,1000)
        distance_z = DerivedProperty("distance_z","Distance between target and airplane in z axis",-250,250)
        
        # Angular states as sin/cos components
        enemy_roll_sin = DerivedProperty("enemy_roll_sin","Enemy roll angle sine component",-1,1)
        enemy_roll_cos = DerivedProperty("enemy_roll_cos","Enemy roll angle cosine component",-1,1)
        enemy_pitch_sin = DerivedProperty("enemy_pitch_sin","Enemy pitch angle sine component",-1,1)
        enemy_pitch_cos = DerivedProperty("enemy_pitch_cos","Enemy pitch angle cosine component",-1,1)
        enemy_heading_sin = DerivedProperty("enemy_heading_sin","Enemy heading angle sine component",-1,1)
        enemy_heading_cos = DerivedProperty("enemy_heading_cos","Enemy heading angle cosine component",-1,1)
        
        # Own aircraft angular states as sin/cos
        own_roll_sin = DerivedProperty("own_roll_sin","Own roll angle sine component",-1,1)
        own_roll_cos = DerivedProperty("own_roll_cos","Own roll angle cosine component",-1,1)
        own_pitch_sin = DerivedProperty("own_pitch_sin","Own pitch angle sine component",-1,1)
        own_pitch_cos = DerivedProperty("own_pitch_cos","Own pitch angle cosine component",-1,1)
        own_heading_sin = DerivedProperty("own_heading_sin","Own heading angle sine component",-1,1)
        own_heading_cos = DerivedProperty("own_heading_cos","Own heading angle cosine component",-1,1)
        
        # 3D LOS error as sin/cos components
        los_azimuth_error_sin = DerivedProperty("los_azimuth_error_sin", "LOS azimuth error sine component",-1,1)
        los_azimuth_error_cos = DerivedProperty("los_azimuth_error_cos", "LOS azimuth error cosine component",-1,1)
        los_elevation_error_sin = DerivedProperty("los_elevation_error_sin", "LOS elevation error sine component",-1,1)
        los_elevation_error_cos = DerivedProperty("los_elevation_error_cos", "LOS elevation error cosine component",-1,1)
        
        # State now includes sin/cos components instead of raw angles
        super().__init__((distance_x, distance_y, distance_z, 
                         own_roll_sin, own_roll_cos, own_pitch_sin, own_pitch_cos, own_heading_sin, own_heading_cos,
                         enemy_roll_sin, enemy_roll_cos, enemy_pitch_sin, enemy_pitch_cos, enemy_heading_sin, enemy_heading_cos,
                         los_azimuth_error_sin, los_azimuth_error_cos, los_elevation_error_sin, los_elevation_error_cos),
                         max_time_s, step_frequency_hz, sim,debug=debug, use_autopilot=True)
        self.aircraft = aircraft
        self.los_azimuth_error_sin_prp = los_azimuth_error_sin
        self.los_azimuth_error_cos_prp = los_azimuth_error_cos
        self.los_elevation_error_sin_prp = los_elevation_error_sin
        self.los_elevation_error_cos_prp = los_elevation_error_cos
        self.distance_prp = distance
        self.visualization_freq = self.step_frequency_hz / 10
        if self.debug:
            self.visualizer = DogfightVisualizer()

    def get_target(self):
        return self.target

    def task_step(self, action: Sequence[float], sim_steps: int) \
        -> Tuple[np.ndarray, float, bool, Dict]:
        if self.debug:
            out_props = [self.los_azimuth_error_sin_prp, self.los_azimuth_error_cos_prp, 
                        self.los_elevation_error_sin_prp, self.los_elevation_error_cos_prp, self.distance_prp]
            for _prop in out_props:
                print(f"{_prop.name} : {self.get_prop(_prop)}")

            print("\n")
        
        self.target.step(1 / self.step_frequency_hz)
        if self.debug:
            if self.current_step % self.visualization_freq == 0:
                self.visualizer.update_from_simulation(self)
                self.visualizer.update_plot()

        return super().task_step(action, sim_steps)

    def get_initial_conditions(self):
        self.target.reset()
        extra_conditions = {prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
                            prp.initial_v_fps: 0,
                            prp.initial_w_fps: 0,
                            prp.initial_p_radps: 0,
                            prp.initial_q_radps: 0,
                            prp.initial_r_radps: 0,
                            prp.initial_roc_fpm: 0,
                            prp.initial_heading_deg: self.INITIAL_HEADING_DEG,
                            }
        self.base_initial_conditions[prp.initial_altitude_ft] = mt2ft(self.GROUND_ALTITUDE_MT + 50 + np.random.random()*75)
        self.base_initial_conditions[prp.initial_latitude_geod_deg] = self.target["Lat"] + (np.random.random()-0.5)*0.004
        self.base_initial_conditions[prp.initial_longitude_geoc_deg] = self.target["Lon"] + (np.random.random()-0.5)*0.004
        return {**self.base_initial_conditions, **extra_conditions}
        
    def _new_episode_init(self):
        super()._new_episode_init()
        self.current_step = 0
        self.sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)
        #self.sim.set_throttle(self.THROTTLE_CMD)
        # Set origin of coordinate system to inital position of airplane
        self.origin = np.array([self.get_prop(_prop) for _prop in [prp.initial_latitude_geod_deg,
                                                                   prp.initial_longitude_geoc_deg,
                                                                   prp.initial_altitude_mt]])
        #TODO: set target plane?

    def is_locked(self):
        if self.get_distance() > 50.0:
            return False
        if abs(self.get_3d_los_azimuth_error()) > 20:
            return False
        if abs(self.get_3d_los_elevation_error()) > 15:  # Added elevation constraint
            return False
        return True
    
    def distance_limit(self):
        return self.get_distance() > 2500
    
    def get_3d_los_azimuth(self):
        """Get azimuth angle of LOS vector in degrees (0-360)"""
        dist = self.get_distance_v()
        # math.atan2 expects (y, x) for azimuth angle
        angle_rad = math.atan2(dist[1], dist[0])
        # Convert radians to degrees
        angle_deg = math.degrees(angle_rad)
        # Normalize the angle to be in the range [0, 360)
        return (90 - angle_deg + 360) % 360
    
    def get_3d_los_elevation(self):
        """Get elevation angle of LOS vector in degrees (-90 to +90)"""
        dist = self.get_distance_v()
        # Calculate horizontal distance
        horizontal_dist = math.sqrt(dist[0]**2 + dist[1]**2)
        # Calculate elevation angle
        elevation_rad = math.atan2(dist[2], horizontal_dist)
        return math.degrees(elevation_rad)
    
    def get_3d_los_azimuth_error(self):
        """Get azimuth error between LOS and aircraft heading"""
        los_azimuth = self.get_3d_los_azimuth()
        heading = self.get_heading()
        
        error = los_azimuth - heading
        
        # Normalize the error to the range [-180, 180]
        if error > 180:
            error -= 360
        elif error < -180:
            error += 360
            
        return error
    
    def get_3d_los_elevation_error(self):
        """Get elevation error between LOS and aircraft pitch"""
        los_elevation = self.get_3d_los_elevation()
        pitch = math.degrees(self.get_prop(prp.pitch_rad))
        
        error = los_elevation - pitch
        
        # Normalize the error to the range [-180, 180]
        if error > 180:
            error -= 360
        elif error < -180:
            error += 360
            
        return error

    def get_heading(self):
        return self.sim[prp.heading_deg]
    
    def altitude_limit(self):
        return self.get_prop(prp.altitude_sl_mt) < 575

    def get_geo_pos(self):
        return np.array([self.get_prop(_prop) for _prop in [prp.lat_geod_deg,
                                                            prp.lng_geoc_deg,
                                                            prp.altitude_sl_mt]])
    def get_target_geo(self):
        return np.array([self.target["Lat"],self.target["Lon"],self.target["Alt"]])

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
        
        # Own aircraft angular states as sin/cos
        if prop.name == "own_roll_sin":
            value = float(math.sin(self.get_prop(prp.roll_rad)))
        if prop.name == "own_roll_cos":
            value = float(math.cos(self.get_prop(prp.roll_rad)))
        if prop.name == "own_pitch_sin":
            value = float(math.sin(self.get_prop(prp.pitch_rad)))
        if prop.name == "own_pitch_cos":
            value = float(math.cos(self.get_prop(prp.pitch_rad)))
        if prop.name == "own_heading_sin":
            value = float(math.sin(math.radians(self.get_heading())))
        if prop.name == "own_heading_cos":
            value = float(math.cos(math.radians(self.get_heading())))
        
        # Enemy aircraft angular states as sin/cos
        if prop.name == "enemy_roll_sin":
            value = float(math.sin(self.target["Roll"]))
        if prop.name == "enemy_roll_cos":
            value = float(math.cos(self.target["Roll"]))
        if prop.name == "enemy_pitch_sin":
            value = float(math.sin(self.target["Pitch"]))
        if prop.name == "enemy_pitch_cos":
            value = float(math.cos(self.target["Pitch"]))
        if prop.name == "enemy_heading_sin":
            value = float(math.sin(math.radians(self.target["Heading"])))
        if prop.name == "enemy_heading_cos":
            value = float(math.cos(math.radians(self.target["Heading"])))
        
        # 3D LOS error as sin/cos
        if prop.name == "los_azimuth_error_sin":
            value = float(math.sin(math.radians(self.get_3d_los_azimuth_error())))
        if prop.name == "los_azimuth_error_cos":
            value = float(math.cos(math.radians(self.get_3d_los_azimuth_error())))
        if prop.name == "los_elevation_error_sin":
            value = float(math.sin(math.radians(self.get_3d_los_elevation_error())))
        if prop.name == "los_elevation_error_cos":
            value = float(math.cos(math.radians(self.get_3d_los_elevation_error())))
            
        return value
        
    def _is_terminal(self,state, sim: Simulation) -> bool:
        done_extra = self.altitude_limit() or self.distance_limit() or self.is_locked()
        return super()._is_terminal(state,sim) or done_extra
    
    def calculate_reward(self, state, done):
        distance = self.get_distance()
        los_azimuth_error = abs(self.get_3d_los_azimuth_error())
        los_elevation_error = abs(self.get_3d_los_elevation_error())
        
        # Base reward starts at 0
        reward = 0.0
        
        # 1. Distance-based reward (encourage getting closer)
        if distance <= 50:
            # High reward for being within engagement range
            reward += 10.0 * (1.0 - distance / 50.0)
        elif distance <= 200:
            # Medium reward for being in intermediate range
            reward += 5.0 * (1.0 - (distance - 50) / 150.0)
        else:
            # Penalty for being too far
            reward -= 0.1 * min((distance - 200) / 100.0, 5.0)
        
        # 2. LOS error penalties (encourage pointing at target)
        azimuth_penalty = -0.1 * min(los_azimuth_error / 10.0, 2.0)
        elevation_penalty = -0.1 * min(los_elevation_error / 10.0, 2.0)
        reward += azimuth_penalty + elevation_penalty
        
        # 3. Bonus for being locked on target
        if self.is_locked():
            reward += 5.0
        
        # 4. Terminal condition rewards/penalties
        if done:
            if self.distance_limit():
                reward -= 20.0  # Big penalty for going too far
            elif self.altitude_limit():
                reward -= 30.0  # Bigger penalty for crashing
            elif self.is_locked():
                reward += 50.0  # Big reward for successful lock
        
        # 5. Small penalty for time (encourage efficiency)
        reward -= 0.01
        
        return reward