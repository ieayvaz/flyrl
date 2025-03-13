from abc import ABC, abstractmethod
import types
from typing import Dict, Sequence, Tuple, Union
from flyrl.autopilot import AutoPilot
from flyrl.tasks import Task
from flyrl.simulation import Simulation
import numpy as np
import flyrl.properties as prp
from flyrl.properties import BoundedProperty, Property, DerivedProperty
from flyrl.aircraft import Aircraft
import math
import gymnasium as gym


class BaseFlightTask(Task, ABC):
    MAXIMUM_ROLL = 55.0
    MAXIMUM_PITCH = 5.0
    INITIAL_ALTITUDE_FT = 2250  #For rascal dogfight. TODO: Make this more flexible
    AUTOPILOT_FREQ = 60.0 # Should not be bigger than sim frequency!
    CRUISE_PITCH = 2.0
    base_state_variables = ()
    base_initial_conditions = {prp.initial_altitude_ft: INITIAL_ALTITUDE_FT,
         prp.initial_terrain_altitude_ft: 0.00000001,
         prp.initial_longitude_geoc_deg: -2.3273,
         prp.initial_latitude_geod_deg: 51.3781  # corresponds to UoBath
         }

    def __init__(self, state_variables, max_time_s, step_frequency_hz, sim : Simulation, debug: bool = False,
                 action_variables : Tuple = (prp.aileron_cmd, prp.elevator_cmd), autopilot=False):
        self.debug = debug
        self.state_variables = state_variables
        self.action_variables = action_variables
        self.max_steps = math.ceil(max_time_s * step_frequency_hz)
        self.sim = sim
        self.step_frequency_hz = step_frequency_hz
        self.use_autopilot = autopilot
        self.current_step = 0
        self.target_roll = 0.0

    def task_step(self, action, sim_steps: int) \
            -> Tuple[np.ndarray, float, bool, Dict]:

        if action == 0 and self.target_roll <= 30:
            self.target_roll += 15.0
        elif action == 1:
            pass
        elif action == 2 and self.target_roll >= -30:
            self.target_roll += -15.0
        _action = None

        if self.use_autopilot:
            _action = self.autopilot.generate_controls(self.target_roll, self.CRUISE_PITCH)
        else:
            _action = action

        for prop, command in zip((prp.aileron_cmd,prp.elevator_cmd), _action):
            self.sim[prop] = command

        # run simulation

        if self.use_autopilot:
            for i in range(sim_steps):
                if i % self.sim.sim_frequency_hz / self.AUTOPILOT_FREQ == 0:
                    _action = self.autopilot.generate_controls(self.target_roll, self.CRUISE_PITCH)
                    for prop, command in zip((prp.aileron_cmd,prp.elevator_cmd), _action):
                        self.sim[prop] = command
                self.sim.run()
        else:
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
        if self.use_autopilot:
            self.autopilot = AutoPilot(self.sim)
            assert self.sim.sim_frequency_hz >= self.AUTOPILOT_FREQ
        self.current_step = 0

    def get_state_space(self) -> gym.Space:
        state_lows = np.array([state_var.min for state_var in self.state_variables])
        state_highs = np.array([state_var.max for state_var in self.state_variables])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype='float')

    def get_action_space(self) -> gym.Space:
        '''
        action_lows = np.array([act_var.min for act_var in self.action_variables])
        action_highs = np.array([act_var.max for act_var in self.action_variables])
        return gym.spaces.Box(low=action_lows, high=action_highs, dtype='float')'
        '''
        #discrete actions
        return gym.spaces.Discrete(3)

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
        
    
    
