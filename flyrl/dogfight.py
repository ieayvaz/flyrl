from math import isnan, pi
import math
from typing import Dict, Sequence, Tuple
import numpy as np
from flyrl.aircraft import Aircraft
from flyrl.basic_tasks import BaseFlightTask
from flyrl.properties import BoundedProperty, DerivedProperty, Property
from flyrl.simulation import Simulation
import flyrl.properties as prp
from flyrl import geoutils
from flyrl.target import Target
from flyrl.utils import angle_between, mt2ft, ft2mt

import matplotlib.pyplot as plt

class DogfightTask(BaseFlightTask):
    INITIAL_HEADING_DEG = 120.0
    THROTTLE_CMD = 0.8 # Default throttle level.
    MIXTURE_CMD = 0.8
    
    def __init__(self, step_frequency_hz: float, sim: Simulation, aircraft: Aircraft, max_time_s: float = 100.0, debug: bool = False):
        self.target = Target("~/flyrl/flight_data2.csv")
        distance_x = DerivedProperty("distance_x","Distance between target and airplane in x axis",-1000,1000)
        distance_y = DerivedProperty("distance_y","Distance between target and airplane in y axis",-1000,1000)
        distance_z = DerivedProperty("distance_z","Distance between target and airplane in z axis",-250,250)
        target_heading_deg = DerivedProperty("target_heading","target heading in degrees",0,360)
        is_locked = DerivedProperty("is_locked","Lock status of airplane",0,1)
        target_roll = DerivedProperty("target_roll_rad","Target roll angle in radians",-pi,pi)
        target_pitch = DerivedProperty("target_pitch_rad","Target pitch angle in radians",-pi/2,pi/2)
        super().__init__((distance_x,distance_y,distance_z,prp.heading_deg,target_roll,target_pitch,
                        prp.roll_rad,prp.pitch_rad),
                         max_time_s, step_frequency_hz, sim,debug=debug)
        self.aircraft = aircraft

    def task_step(self, action: Sequence[float], sim_steps: int) \
        -> Tuple[np.ndarray, float, bool, Dict]:
        if self.debug:
            out_props = [prp.sim_time_s, prp.altitude_sl_mt, prp.alpha_deg, prp.v_east_fps, prp.v_north_fps, prp.v_down_fps]
            for _prop in out_props:
                print(f"{_prop.name} : {self.get_prop(_prop)}")

            print("\n")
        self.target.step(0.1)
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
        self.base_initial_conditions[prp.initial_altitude_ft] = mt2ft(75 + np.random.random()*75)
        self.base_initial_conditions[prp.initial_latitude_geod_deg] = self.target["Lat"] + (np.random.random()-0.5)*0.01
        self.base_initial_conditions[prp.initial_longitude_geoc_deg] = self.target["Lon"] + (np.random.random()-0.5)*0.01
        return {**self.base_initial_conditions, **extra_conditions}
        
    def _new_episode_init(self):
        super()._new_episode_init()
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
        dist_azimuth = math.atan2(self.get_distance_v()[0],self.get_distance_v()[1]) % 360
        if abs(self.sim[prp.heading_deg] - dist_azimuth) > 20.0:
            return False
        return True

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
        return (prp.u_fps,prp.altitude_sl_ft,prp.lat_geod_deg,prp.lng_geoc_deg,prp.heading_deg)

    def get_distance(self):
        return np.linalg.norm(self.get_target_pos() - self.get_pos())

    def get_distance_v(self):
        return self.get_target_pos() - self.get_pos() 

    def get_prop(self, prop) -> float:
        value = super().get_prop(prop)
        if value != None:
            return value
        # Process derived properties  
        if prop.name == "distance_x":
            value = float(self.get_distance_v()[0])
        if prop.name == "distance_y":
            value = float(self.get_distance_v()[1])
        if prop.name == "distance_z":
            value = float(self.get_distance_v()[2])
        '''if prop.name == "target_heading":
            return self.target["Heading"]'''
        if prop.name == "target_roll_rad":
            value = float(self.target["Roll"])
        if prop.name == "target_pitch_rad":
            value = float(self.target["Pitch"])
        return value
        
    def _is_terminal(self,state, sim: Simulation) -> bool:
        term_cond = self.is_locked()
        return super()._is_terminal(state,sim) or term_cond
    
    def calculate_reward(self, state):
        SCALE = 2.0
        if self.is_locked() == True:
            return 10*SCALE
        if self._is_terminal(state.values(),self.sim) == True:
            return -0.1*SCALE
        rew = 0
        if self.sim[prp.roll_rad] > np.pi/3.0:
            rew -= 0.1*SCALE
        if self.sim[prp.pitch_rad] > np.pi/3.0:
            rew -= 0.1*SCALE
        
        rew += (1 - self.get_distance()/500.0) *SCALE

        return rew