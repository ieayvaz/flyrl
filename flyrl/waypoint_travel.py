from math import isnan
import numpy as np
from flyrl.aircraft import Aircraft
from flyrl.basic_tasks import BaseFlightTask
from flyrl.properties import BoundedProperty, DerivedProperty, Property
from flyrl.simulation import Simulation
import flyrl.properties as prp
from flyrl import geoutils
from flyrl.utils import angle_between

import matplotlib.pyplot as plt

class WaypointTask(BaseFlightTask):
    INITIAL_HEADING_DEG = 120.0
    MAX_DISTANCE = 1500 # Max distance in meters
    THROTTLE_CMD = 0.8 # Default throttle level. Throttle not changed during this task
    MIXTURE_CMD = 0.8
    
    def __init__(self, step_frequency_hz: float, sim: Simulation, aircraft: Aircraft, max_time_s: float = 100.0, debug : bool = False):
        self.position = [DerivedProperty("pos_x","Position along x axis",-1000,1000),
                         DerivedProperty("pos_y","Position along y axis",-1000,1000),
                         DerivedProperty("pos_z","Position along z axis",0,1000)]
        self.waypoint_position = [DerivedProperty("wpos_x","Waypoint position along x axis",-1000,1000),
                         DerivedProperty("wpos_y","Waypoint position along y axis",-1000,1000),
                         DerivedProperty("wpos_z","Waypoint position along z axis",0,1000)]
        distance = DerivedProperty("distance","Distance between waypoint and airplane",0,3000)
        delta_heading = DerivedProperty("delta_h","Diffrence between heading and waypoint direction",0,360)
        super().__init__((delta_heading,prp.u_fps,prp.roll_rad,prp.pitch_rad,distance),
                         max_time_s, step_frequency_hz, sim,debug=debug)
        self.aircraft = aircraft
        if self.debug:
            plt.figure()
            self.positions_x = []
            self.positions_y = []

    def get_initial_conditions(self):
        extra_conditions = {prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
                            prp.initial_v_fps: 0,
                            prp.initial_w_fps: 0,
                            prp.initial_p_radps: 0,
                            prp.initial_q_radps: 0,
                            prp.initial_r_radps: 0,
                            prp.initial_roc_fpm: 0,
                            prp.initial_heading_deg: self.INITIAL_HEADING_DEG,
                            }
        return {**self.base_initial_conditions, **extra_conditions}
        
    def _new_episode_init(self):
        super()._new_episode_init()
        self.sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)
        # Set origin of coordinate system to inital position of airplane
        self.origin = np.array([self.get_prop(_prop) for _prop in [prp.initial_latitude_geod_deg,
                                                                   prp.initial_longitude_geoc_deg,
                                                                   prp.initial_altitude_mt]])
        # Assign a rondom waypoint for episode.
        # future work can get this from an outside source
        self.waypoint_pos = 2*(np.random.rand(3)-0.5)*np.array([400,400,500])

    def get_geo_pos(self):
        return np.array([self.get_prop(_prop) for _prop in [prp.lat_geod_deg,
                                                            prp.lng_geoc_deg,
                                                            prp.altitude_sl_mt]])

    def get_heading_error(self):
        dist = (self.waypoint_pos - self.get_pos())[:2]
        target_angle = (np.arctan2(dist[0],dist[1])*180.0/np.pi + 360) % 360
        if self.debug:
            plt.clf()
            self.positions_x.append(self.waypoint_pos[0])
            self.positions_x.append(self.get_pos()[0])
            self.positions_y.append(self.waypoint_pos[1])
            self.positions_y.append(self.get_pos()[1])
            plt.scatter(self.positions_x,self.positions_y)
            plt.plot([self.waypoint_pos[0],self.get_pos()[0]],[self.waypoint_pos[1],self.get_pos()[1]])
            plt.pause(0.1)
        heading_ang = self.sim[prp.heading_deg]
        delta = heading_ang - target_angle
        comp_delta = 360 + delta if delta < 0 else 360 - delta
        if abs(delta) < abs(comp_delta):
            return delta
        else:
            return comp_delta

    def get_pos(self):
        return geoutils.lla_2_enu(self.get_geo_pos(),self.origin)

    def get_props_to_output(self):
        return (prp.u_fps,prp.altitude_sl_ft,prp.lat_geod_deg,prp.lng_geoc_deg,prp.heading_deg)

    def get_distance(self):
        return np.linalg.norm(self.waypoint_pos - self.get_pos()) 

    def get_prop(self, prop) -> float:
        native = super().get_prop(prop)
        if native != None:
            return native
        # Process derived properties  
        if prop.name == "distance":
            return self.get_distance()
        if prop.name == "delta_h":
            return self.get_heading_error()
        elif prop in self.position:
            return dict(zip([_prop.name for _prop in self.position],self.get_pos()))[prop.name]
        elif prop in self.waypoint_position:
            return dict(zip([_prop.name for _prop in self.waypoint_position],self.waypoint_pos))[prop.name] 
        
    def _is_terminal(self,state, sim: Simulation) -> bool:
        return super()._is_terminal(state,sim)
    
    def calculate_reward(self, state):
        SCALE = 2.0
        if self._is_terminal(state.values(),self.sim) == True:
            return -0.1*SCALE
        rew = 0
        '''
        rew += 1 - state["delta_h"]/180.0
        rew += -0.05

        if state["distance"] < 100.0:
            rew += 0.5
        if state["distance"] < 20.0:
            rew += 5.0
        '''
        if self.sim[prp.roll_rad] > np.pi/3.0:
            rew -= 0.1*SCALE
        if self.sim[prp.pitch_rad] > np.pi/3.0:
            rew -= 0.1*SCALE
        
        rew += (1 - state["delta_h"]/180.0) *SCALE
        #rew += (1 - state["distance"]/1000.0) *SCALE

        if self.debug:
            print(f"Reward: {rew} State: {state}")
        return rew
        