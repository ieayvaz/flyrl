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