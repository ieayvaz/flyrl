from abc import ABC, abstractmethod
import types
from typing import Dict, Sequence, Tuple, Union
from flyrl.tasks import Task
from flyrl.simulation import Simulation
import numpy as np
import flyrl.properties as prp
from flyrl.properties import BoundedProperty, Property, DerivedProperty
from flyrl.aircraft import Aircraft
import math
import gymnasium as gym


class BaseFlightTask(Task, ABC):
    INITIAL_ALTITUDE_FT = 5000
    base_state_variables = ()
    base_initial_conditions = types.MappingProxyType(  # MappingProxyType makes dict immutable
        {prp.initial_altitude_ft: INITIAL_ALTITUDE_FT,
         prp.initial_terrain_altitude_ft: 0.00000001,
         prp.initial_longitude_geoc_deg: -2.3273,
         prp.initial_latitude_geod_deg: 51.3781  # corresponds to UoBath
         }
    )

    def __init__(self, state_variables, max_time_s, step_frequency_hz, sim : Simulation, debug: bool = False):
        self.debug = debug
        self.state_variables = state_variables
        self.action_variables = (prp.aileron_cmd, prp.elevator_cmd)
        self.max_steps = math.ceil(max_time_s * step_frequency_hz)
        self.sim = sim
        self.current_step = 0

    def task_step(self, action: Sequence[float], sim_steps: int) \
            -> Tuple[np.ndarray, float, bool, Dict]:

        for prop, command in zip(self.action_variables, action):
            self.sim[prop] = command

        # run simulation
        for _ in range(sim_steps):
            self.sim.run()
            
        state = self.get_state()
        done = self._is_terminal(state, self.sim)
        reward = self.calculate_reward(dict(zip([prop.name for prop in self.state_variables],state)))

        self.current_step += 1

        return state, reward, done, {}
    
    def observe_first_state(self) -> np.ndarray:
        self._new_episode_init()
        state = self.get_state()
        return state
    
    def get_state(self):
        state = [self.get_prop(prop) for prop in self.state_variables]
        return state
    
    def set_sim(self, sim):
        self.sim = sim
        
    @abstractmethod
    def get_props_to_output(self) -> Tuple:
        ...
    
    def _new_episode_init(self):
        self.sim.start_engines()
        self.sim.raise_landing_gear()
        self.current_step = 0

    def get_state_space(self) -> gym.Space:
        state_lows = np.array([state_var.min for state_var in self.state_variables])
        state_highs = np.array([state_var.max for state_var in self.state_variables])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype='float')

    def get_action_space(self) -> gym.Space:
        action_lows = np.array([act_var.min for act_var in self.action_variables])
        action_highs = np.array([act_var.max for act_var in self.action_variables])
        return gym.spaces.Box(low=action_lows, high=action_highs, dtype='float')

    @abstractmethod
    def get_initial_conditions(self) -> Dict[Property, float]:
        ...

    @abstractmethod     
    def get_prop(self, prop) -> float:
        if type(prop) == BoundedProperty or type(prop) == Property:
            return self.sim[prop]
        if prop.name == "h-sl-mt":
            return self.sim[prp.altitude_sl_ft]*0.3048
        if prop.name == "ic/h-sl-mt":
            return self.sim[prp.initial_altitude_ft]*0.3048
    
        return None

    @abstractmethod
    def _is_terminal(self, state, sim: Simulation) -> bool:
        for _state in state:
            if math.isnan(_state):
                return True
        if self.max_steps - self.current_step <= 0:
            return True
        return False

    @abstractmethod
    def calculate_reward(self,state) -> float:
        ...

class TaskHeading(BaseFlightTask):
    INITIAL_HEADING_DEG = 120.0
    THROTTLE_CMD = 0.8
    MIXTURE_CMD = 0.8
    
    def __init__(self, step_frequency_hz: float, sim: Simulation, aircraft: Aircraft, max_time_s: float = 60.0):
        delta_heading = DerivedProperty("delta_heading","Diffrence between current and target heading angles",0,360)
        super().__init__((delta_heading,),max_time_s, step_frequency_hz, sim)
        self.target_heading = 90
        self.aircraft = aircraft

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
    
    def get_heading_error(self):
        return abs(self.sim[prp.heading_deg] - self.target_heading)

    def get_props_to_output(self):
        return (prp.u_fps,prp.altitude_sl_ft)
    
    def get_prop(self, prop,sim):
        native = super().get_prop(prop)
        if native != None:
            return native
        if prop.name == "delta_heading":
            return self.get_heading_error()
        
    def _is_terminal(self,state, sim: Simulation) -> bool:
        if self.max_steps - self.current_step <= 0:
            return True
        return False
    
    def calculate_reward(self, state):
        return 1 - state["delta_heading"] / 180.0
        
    
    
