from typing import Any, Dict, Tuple, Type
import gymnasium as gym
import numpy as np
from flyrl.ap_simulation import AP_Simulation
from flyrl.ap_tasks import BaseAPTask
from flyrl.aircraft import Aircraft
from flyrl.simulation import Simulation
from flyrl.visualiser import FigureVisualiser, FlightGearRemoteVisualiser

class APEnv(gym.Env):
    JSBSIM_DT_HZ = 120
    metadata = {'render.modes': ['human']}

    def __init__(self, task_type: Type[BaseAPTask], aircraft: Aircraft, ap_address : str = '172.31.6.43:14550', enemy_address: str = '172.31.6.43:14560',
                  agent_interaction_freq: int = 10, debug : bool = False):
        if agent_interaction_freq > self.JSBSIM_DT_HZ:
            raise ValueError('agent interaction frequency must be less than '
                             'or equal to JSBSim integration frequency of '
                             f'{self.JSBSIM_DT_HZ} Hz.')
        self.debug = debug
        self.sim: Simulation = None
        self.enemy_sim: Simulation = None
        self.sim_steps_per_agent_step: int = self.JSBSIM_DT_HZ // agent_interaction_freq
        self.aircraft = aircraft
        self.task = task_type(step_frequency_hz=agent_interaction_freq,player_sim=None,enemy_sim=None,aircraft=aircraft, debug=debug)
        self.observation_space: gym.spaces.Box = self.task.get_state_space()
        self.action_space: gym.spaces.Box = self.task.get_action_space()
        self.figure_visualiser: FigureVisualiser = None
        self.flightgear_visualiser: FlightGearRemoteVisualiser = None
        self.step_delay = None
        self.ap_address = ap_address
        self.enemy_address = enemy_address

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        state, reward, done, info = self.task.task_step(action, self.sim_steps_per_agent_step)

        return np.array(state), reward, done, done, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed,options=options)
        self.sim = self._init_new_sim(self.ap_address)
        self.enemy_sim = self._init_enemy_sim(self.enemy_address)

        self.task.set_sim(self.sim)
        self.task.set_enemy_sim(self.enemy_sim)
        state = self.task.observe_first_state()

        if self.flightgear_visualiser:
            self.flightgear_visualiser.configure_simulation_output(self.sim)
        return np.array(state), {}
    
    def _init_new_sim(self, address):
            return AP_Simulation(address)
    
    def _init_enemy_sim(self, address):
            return AP_Simulation(address, controlled=False)
    
    def render(self, mode='human', flightgear_blocking=True):
            if mode == 'human':
                pass
            elif mode == 'flightgear':
                if not self.flightgear_visualiser:
                    self.flightgear_visualiser = FlightGearRemoteVisualiser(self.sim,
                                                                    self.task.get_props_to_output(),
                                                                    flightgear_blocking)
                self.flightgear_visualiser.plot(self.sim)
            else:
                super().render(mode=mode)

    def close(self):
        if self.sim:
            self.sim.close()
        if self.figure_visualiser:
            self.figure_visualiser.close()
        if self.flightgear_visualiser:
            self.flightgear_visualiser.close()

    def seed(self, seed=None):  
        gym.logger.warn("Could not seed environment %s", self)
        return