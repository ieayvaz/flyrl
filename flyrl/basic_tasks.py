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
    MAXIMUM_ROLL = 65.0
    MAXIMUM_PITCH = 25.0
    INITIAL_ALTITUDE_FT = 2250  #For rascal dogfight. TODO: Make this more flexible
    AUTOPILOT_FREQ = 60.0 # Should not be bigger than sim frequency!
    CRUISE_PITCH = 2.0
    ROLL_INCREMENT = 15.0
    PITCH_INCREMENT = 5.0
    THROTTLE_INCREMENT = 0.1
    base_state_variables = ()
    base_initial_conditions = {prp.initial_altitude_ft: INITIAL_ALTITUDE_FT,
         prp.initial_terrain_altitude_ft: 0.00000001,
         prp.initial_longitude_geoc_deg: -2.3273,
         prp.initial_latitude_geod_deg: 51.3781  # corresponds to UoBath
         }

    def __init__(self, state_variables, max_time_s, step_frequency_hz, sim : Simulation, debug: bool = False,
                 action_variables : Tuple = (prp.aileron_cmd, prp.elevator_cmd), use_autopilot=False):
        self.debug = debug
        self.state_variables = state_variables
        self.action_variables = action_variables
        self.max_steps = math.ceil(max_time_s * step_frequency_hz)
        self.sim = sim
        self.step_frequency_hz = step_frequency_hz
        self.use_autopilot = use_autopilot
        self.current_step = 0
        self.target_roll = 0.0
        self.target_pitch = 2.0
        self.target_throttle = 0.8

    def task_step(self, action, sim_steps: int) \
            -> Tuple[np.ndarray, float, bool, Dict]:
        
        roll_action, pitch_action, throttle_action = action

        if roll_action == 0:  # Decrease Roll (Bank Left)
            self.target_roll -= self.ROLL_INCREMENT
        elif roll_action == 2:  # Increase Roll (Bank Right)
            self.target_roll += self.ROLL_INCREMENT
        
        # Update Target Pitch
        if pitch_action == 0:  # Decrease Pitch (Nose Down)
            self.target_pitch -= self.PITCH_INCREMENT
        elif pitch_action == 2:  # Increase Pitch (Nose Up)
            self.target_pitch += self.PITCH_INCREMENT

        # Update Target Throttle
        if throttle_action == 0:  # Decrease Throttle
            self.target_throttle -= self.THROTTLE_INCREMENT
        elif throttle_action == 2:  # Increase Throttle
            self.target_throttle += self.THROTTLE_INCREMENT

        # Clip the target values to stay within safe/realistic limits
        self.target_roll = np.clip(self.target_roll, -self.MAXIMUM_ROLL, self.MAXIMUM_ROLL)
        self.target_pitch = np.clip(self.target_pitch, -self.MAXIMUM_PITCH ,self.MAXIMUM_PITCH)
        self.target_throttle = np.clip(self.target_throttle, 0.5, 1.0)

        if self.use_autopilot:
            _action = self.autopilot.generate_controls(self.target_roll, self.target_pitch)
        else:
            _action = (0,0)

        for prop, command in zip((prp.aileron_cmd,prp.elevator_cmd), _action):
            self.sim[prop] = command
            self.sim[prp.throttle_cmd] = self.target_throttle

        # run simulation

        if self.use_autopilot:
            for i in range(sim_steps):
                if i % self.autopilot_update_interval:
                    _action = self.autopilot.generate_controls(self.target_roll, self.CRUISE_PITCH)
                    for prop, command in zip((prp.aileron_cmd,prp.elevator_cmd), _action):
                        self.sim[prop] = command
                self.sim.run()
        else:
            for _ in range(sim_steps):
                self.sim.run()
            
        state = self.get_state()
        done = self._is_terminal(state, self.sim)

        reward = self.calculate_reward(dict(zip([prop.name for prop in self.state_variables],state)), done)
        state_norm = self.get_state_norm()
        self.current_step += 1

        return state_norm, reward, done, {}
    
    def observe_first_state(self) -> np.ndarray:
        self._new_episode_init()
        state = self.get_state()
        return state
    
    def get_state(self):
        state = [self.get_prop(prop) for prop in self.state_variables]
        return state
    
    def get_state_norm(self):
        state = [(np.clip(self.get_prop(prop),prop.min,prop.max)-prop.min)/(prop.max - prop.min) for prop in self.state_variables]
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
            self.autopilot_update_interval = self.sim.sim_frequency_hz // self.AUTOPILOT_FREQ
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
        return gym.spaces.MultiDiscrete([3, 3, 3])

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
                print("NaN Error")
                exit()
        if self.max_steps - self.current_step <= 0:
            return True
        return False

    @abstractmethod
    def calculate_reward(self,state) -> float:
        ...
        
    
    
