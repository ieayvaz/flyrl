from typing import Any, Dict, Tuple, Type
import gymnasium as gym
import numpy as np
from flyrl.multiaircraft_tasks import MultiAircraftFlightTask
from flyrl.aircraft import Aircraft
from flyrl.simulation import Simulation
from flyrl.visualiser import FigureVisualiser, FlightGearRemoteVisualiser

class MultiAircraftJsbSimEnv(gym.Env):
    JSBSIM_DT_HZ: int = 120  # JSBSim integration frequency
    metadata = {'render.modes': ['human', 'flightgear']}

    def __init__(self, task_type: Type[MultiAircraftFlightTask], 
                 player_aircraft: Aircraft, enemy_aircraft: Aircraft,
                 agent_interaction_freq: int = 20, debug: bool = False):
        """
        Initialize multi-aircraft environment
        
        Args:
            task_type: The multi-aircraft task class (e.g., MultiAircraftDogfightTask)
            player_aircraft: Aircraft instance for the player
            enemy_aircraft: Aircraft instance for the enemy
            agent_interaction_freq: How often the agent acts (Hz)
            debug: Enable debug mode
        """
        if agent_interaction_freq > self.JSBSIM_DT_HZ:
            raise ValueError('agent interaction frequency must be less than '
                             'or equal to JSBSim integration frequency of '
                             f'{self.JSBSIM_DT_HZ} Hz.')
        
        self.debug = debug
        self.player_sim: Simulation = None
        self.enemy_sim: Simulation = None
        self.sim_steps_per_agent_step: int = self.JSBSIM_DT_HZ // agent_interaction_freq
        self.player_aircraft = player_aircraft
        self.enemy_aircraft = enemy_aircraft
        self.agent_interaction_freq = agent_interaction_freq
        
        # Initialize task with None simulations (will be set in reset)
        self.task = task_type(
            step_frequency_hz=agent_interaction_freq,
            player_sim=None,
            enemy_sim=None,
            player_aircraft=player_aircraft,
            enemy_aircraft=enemy_aircraft,
            
            debug=debug
        )
        
        # Set up spaces
        self.observation_space: gym.spaces.Box = self.task.get_state_space()
        self.action_space: gym.spaces.Box = self.task.get_action_space()
        
        # Visualization
        self.figure_visualiser: FigureVisualiser = None
        self.flightgear_visualiser: FlightGearRemoteVisualiser = None
        self.step_delay = None
        
        # Initialize environment
        self.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Run one timestep of the environment's dynamics.
        
        Args:
            action: the agent's action for the player aircraft
            
        Returns:
            observation: agent's observation of the current environment
            reward: amount of reward returned after previous action
            terminated: whether the episode has ended
            truncated: whether the episode was truncated
            info: auxiliary information
        """
        # Validate action shape
        if hasattr(self.action_space, 'shape') and action.shape != self.action_space.shape:
            if hasattr(self.action_space, 'nvec'):  # MultiDiscrete
                expected_shape = (len(self.action_space.nvec),)
            else:
                expected_shape = self.action_space.shape
            if action.shape != expected_shape:
                raise ValueError(f'Action shape {action.shape} does not match '
                               f'action space shape {expected_shape}')

        # Execute task step
        state, reward, done, info = self.task.task_step(action, self.sim_steps_per_agent_step)

        # Return in new gymnasium format (terminated, truncated)
        return np.array(state), reward, done, False, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """
        Reset the environment and return initial observation.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observation: Initial observation
            info: Additional info
        """
        super().reset(seed=seed, options=options)
        
        # Get initial conditions for both aircraft
        player_init_conditions = self.task.get_player_initial_conditions()
        enemy_init_conditions = self.task.get_enemy_initial_conditions()
        
        # Initialize or reinitialize simulations
        if self.player_sim:
            self.player_sim.close()
        if self.enemy_sim:
            self.enemy_sim.close()
            
        self.player_sim = self._init_new_sim(
            self.JSBSIM_DT_HZ, self.player_aircraft, player_init_conditions
        )
        self.enemy_sim = self._init_new_sim(
            self.JSBSIM_DT_HZ, self.enemy_aircraft, enemy_init_conditions
        )
        
        # Set simulations in task
        self.task.player_sim = self.player_sim
        self.task.enemy_sim = self.enemy_sim
        
        # Get initial state
        state = self.task.observe_first_state()
        
        # Configure visualization if needed
        if self.flightgear_visualiser:
            self.flightgear_visualiser.configure_simulation_output(self.player_sim)
            
        return np.array(state), {}
    
    def _init_new_sim(self, dt: int, aircraft: Aircraft, initial_conditions: Dict):
        """Initialize a new simulation instance"""
        return Simulation(
            sim_frequency_hz=dt,
            aircraft=aircraft,
            init_conditions=initial_conditions
        )
    
    def render(self, mode='flightgear', flightgear_blocking=True):
        """
        Render the environment.
        
        Args:
            mode: Rendering mode ('human' or 'flightgear')
            flightgear_blocking: Whether to wait for FlightGear to load
        """
        if mode == 'human':
            if not self.figure_visualiser:
                self.figure_visualiser = FigureVisualiser(
                    self.player_sim,
                    self.task.get_props_to_output()
                )
            self.figure_visualiser.plot(self.player_sim)
            
        elif mode == 'flightgear':
            if not self.flightgear_visualiser:
                self.flightgear_visualiser = FlightGearRemoteVisualiser(
                    self.player_sim,
                    self.task.get_props_to_output(),
                    flightgear_blocking
                )
            self.flightgear_visualiser.plot(self.player_sim)
            
        else:
            super().render(mode=mode)

    def close(self):
        """Clean up environment resources"""
        if self.player_sim:
            self.player_sim.close()
        if self.enemy_sim:
            self.enemy_sim.close()
        if self.figure_visualiser:
            self.figure_visualiser.close()
        if self.flightgear_visualiser:
            self.flightgear_visualiser.close()

    def seed(self, seed=None):
        """
        Set the seed for this environment's random number generator(s).
        
        Args:
            seed: Random seed value
            
        Returns:
            List of seeds used in this environment's random number generators
        """
        gym.logger.warn("Could not seed environment %s", self)
        return
        
    def get_player_state(self):
        """Get the current state of the player aircraft"""
        if self.player_sim:
            return self.task.get_player_state()
        return None
        
    def get_enemy_state(self):
        """Get the current state of the enemy aircraft"""
        if self.enemy_sim:
            return self.task.get_enemy_state()
        return None
        
    def get_relative_state(self):
        """Get relative state information between aircraft"""
        return self.task.get_relative_state()
        
    def get_distance_between_aircraft(self):
        """Get distance between player and enemy aircraft"""
        if hasattr(self.task, 'get_distance'):
            return self.task.get_distance()
        return None
        
    def is_episode_locked(self):
        """Check if player has locked onto enemy"""
        if hasattr(self.task, 'is_locked'):
            return self.task.is_locked()
        return False