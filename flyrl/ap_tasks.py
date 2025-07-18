from abc import ABC, abstractmethod
import types
from typing import Dict, Sequence, Tuple, Union, List
from flyrl.autopilot import AutoPilot
from flyrl.tasks import Task
from flyrl.simulation import Simulation
import numpy as np
import flyrl.properties as prp
from flyrl.properties import BoundedProperty, Property, DerivedProperty
from flyrl.aircraft import Aircraft
import math
import gymnasium as gym

class BaseAPTask(Task, ABC):
    MAXIMUM_ROLL = 60.0
    MAXIMUM_PITCH = 25.0
    INITIAL_ALTITUDE_FT = 2250
    AUTOPILOT_FREQ = 60.0
    CRUISE_PITCH = 2.0
    ROLL_INCREMENT = 15.0
    PITCH_INCREMENT = 5.0
    THROTTLE_INCREMENT = 0.1
    
    # Action space limits - normalized to [-1, 1]
    ROLL_MIN = -MAXIMUM_ROLL      # -60 degrees
    ROLL_MAX = MAXIMUM_ROLL       # +60 degrees
    PITCH_MIN = -MAXIMUM_PITCH    # -25 degrees
    PITCH_MAX = MAXIMUM_PITCH     # +25 degrees
    THROTTLE_MIN = 0.0
    THROTTLE_MAX = 1.0
    
    base_state_variables = ()
    base_initial_conditions = {
        prp.initial_altitude_ft: INITIAL_ALTITUDE_FT,
        prp.initial_terrain_altitude_ft: 0.00000001,
        prp.initial_longitude_geoc_deg: -2.3273,
        prp.initial_latitude_geod_deg: 51.3781
    }

    def __init__(self, state_variables, max_time_s, step_frequency_hz, 
                 player_sim: Simulation,
                 enemy_sim: Simulation,
                 debug: bool = False,
                 action_variables: Tuple = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd, prp.throttle_cmd), 
                 use_autopilot=False,):
        self.debug = debug
        self.state_variables = state_variables
        self.action_variables = action_variables
        self.max_steps = math.ceil(max_time_s * step_frequency_hz)
        
        # Separate simulations for player and enemy
        self.player_sim = player_sim
        self.enemy_sim = enemy_sim
        
        self.step_frequency_hz = step_frequency_hz
        self.use_autopilot = use_autopilot
        
        self.current_step = 0
        
        # Player aircraft control targets
        self.player_target_roll = 0
        self.player_target_pitch = 2.0
        self.player_target_throttle = 1.0

    def task_step(self, action, sim_steps: int, visualizer, env) -> Tuple[np.ndarray, float, bool, Dict]:
        # Process player action (normalized inputs)
        self._process_player_action(action)
        print(f"Action: {action}")
        
        # Apply controls to both aircraft
        self._apply_player_controls(action)
        
        # Run both simulations
        self._run_simulations(sim_steps, action, visualizer, env)
        
        # Get combined state
        state = self.get_state()
        done = self._is_terminal(state)
        
        reward = self.calculate_reward(
            dict(zip([prop.name for prop in self.state_variables], state)), 
            done
        )
        
        state_norm = self.get_state_norm()
        self.current_step += 1
        info_dict = {"success" : self.get_episode_success()}
        
        return state_norm, reward, done, info_dict
    
    def _process_player_action(self, action):
        """
        Process the player's normalized action:
        action[0]: Roll command (normalized -1 to 1)
        action[1]: Pitch command (normalized -1 to 1)
        action[2]: Throttle command (normalized -1 to 1)
        """
        # Denormalize actions from [-1, 1] to actual command ranges
        self.player_target_roll = self._denormalize_action(
            action[0], self.ROLL_MIN, self.ROLL_MAX
        )
        self.player_target_pitch = self._denormalize_action(
            action[1], self.PITCH_MIN, self.PITCH_MAX
        )
        self.player_target_throttle = self._denormalize_action(
            action[2], self.THROTTLE_MIN, self.THROTTLE_MAX
        )
    
    def _denormalize_action(self, normalized_action, min_val, max_val):
        """
        Convert normalized action from [-1, 1] to actual command range [min_val, max_val]
        """
        # Clamp normalized action to [-1, 1]
        normalized_action = np.clip(normalized_action, -1.0, 1.0)
        
        # Convert from [-1, 1] to [min_val, max_val]
        return min_val + (normalized_action + 1.0) * (max_val - min_val) / 2.0
    
    def _normalize_action(self, action, min_val, max_val):
        """
        Convert action from [min_val, max_val] to normalized range [-1, 1]
        """
        # Clamp action to valid range
        action = np.clip(action, min_val, max_val)
        
        # Convert from [min_val, max_val] to [-1, 1]
        return 2.0 * (action - min_val) / (max_val - min_val) - 1.0
    
    def _apply_player_controls(self, action = (0,0)):
        """Apply controls to player aircraft using autopilot with roll/pitch commands"""
        if self.use_autopilot:
            # Generate control surface commands via autopilot
            _action = self.player_autopilot.generate_controls(
                self.player_target_roll, 
                self.player_target_pitch
            )
        else:
            # Direct control - you would implement direct control mapping here
            _action = (action[0], action[1])

        # Apply control surface commands
        for prop, command in zip((prp.aileron_cmd, prp.elevator_cmd), _action):
            self.player_sim[prop] = command
        
        # Apply throttle directly
        self.player_sim[prp.throttle_cmd] = self.player_target_throttle
    
    def _run_simulations(self, sim_steps, player_action, visualizer, env):
        """Run both simulations synchronously"""
        if self.use_autopilot:
            for i in range(sim_steps):
                if i % self.player_autopilot_update_interval == 0:
                    _action = self.player_autopilot.generate_controls(
                        self.player_target_roll, 
                        self.player_target_pitch
                    )
                    for prop, command in zip((prp.aileron_cmd, prp.elevator_cmd), _action):
                        self.player_sim[prop] = command
                    self.player_sim[prp.throttle_cmd] = self.player_target_throttle
                self.player_sim.run(visualizer)
        else:
            for _ in range(sim_steps):
                self.player_sim.run(visualizer, env)

    def get_action_space(self) -> gym.Space:
        """
        Return normalized action space [-1, 1] for all actions:
        - action[0]: Roll command (maps to -60 to +60 degrees)
        - action[1]: Pitch command (maps to -25 to +25 degrees)  
        - action[2]: Throttle command (maps to 0 to 1)
        """
        return gym.spaces.Box(
            low=np.array([-1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
    
    def get_denormalized_action_info(self):
        """
        Return information about how normalized actions map to actual commands
        """
        return {
            'roll': {'min': self.ROLL_MIN, 'max': self.ROLL_MAX, 'units': 'degrees'},
            'pitch': {'min': self.PITCH_MIN, 'max': self.PITCH_MAX, 'units': 'degrees'},
            'throttle': {'min': self.THROTTLE_MIN, 'max': self.THROTTLE_MAX, 'units': 'fraction'}
        }

    # ... rest of the methods remain the same ...

    def _get_player_speed_mps(self) -> float:
        """Get player aircraft speed in m/s"""
        # Get velocity components in body frame (fps)
        u_fps = self.player_sim[prp.u_fps]
        v_fps = self.player_sim[prp.v_fps]
        w_fps = self.player_sim[prp.w_fps]
        
        # Calculate total speed in fps, then convert to m/s
        speed_fps = math.sqrt(u_fps**2 + v_fps**2 + w_fps**2)
        speed_mps = speed_fps * 0.3048  # Convert fps to m/s
        
        return speed_mps
    
    def _get_enemy_relative_info(self) -> Tuple[float, float, float]:
        """
        Get enemy relative information: bearing, elevation, and range.
        This is a default implementation that should be overridden by subclasses
        for more accurate calculations.
        """
        # Get positions (this is a simplified implementation)
        player_pos = self.get_player_position()
        enemy_pos = self.get_enemy_position()
        
        # Calculate relative vector
        dx = enemy_pos[0] - player_pos[0]  # longitude difference
        dy = enemy_pos[1] - player_pos[1]  # latitude difference
        dz = enemy_pos[2] - player_pos[2]  # altitude difference
        
        # Convert to approximate distances (simplified)
        # Note: This is a rough approximation - subclasses should override with proper geo calculations
        dx_m = dx * 111000 * math.cos(math.radians(player_pos[1]))  # longitude to meters
        dy_m = dy * 111000  # latitude to meters
        dz_m = dz * 0.3048  # feet to meters
        
        # Calculate bearing (from north, clockwise)
        bearing_rad = math.atan2(dx_m, dy_m)
        bearing_deg = math.degrees(bearing_rad)
        
        # Calculate elevation angle
        horizontal_dist = math.sqrt(dx_m**2 + dy_m**2)
        elevation_rad = math.atan2(dz_m, horizontal_dist)
        elevation_deg = math.degrees(elevation_rad)
        
        # Calculate range
        range_m = math.sqrt(dx_m**2 + dy_m**2 + dz_m**2)
        
        # Convert bearing to relative to aircraft heading
        player_heading = self.player_sim[prp.heading_deg]
        relative_bearing = bearing_deg - player_heading
        
        # Normalize to [-180, 180]
        while relative_bearing > 180:
            relative_bearing -= 360
        while relative_bearing < -180:
            relative_bearing += 360
        
        return relative_bearing, elevation_deg, range_m
    
    def _get_relative_velocity_mps(self) -> np.ndarray:
        """
        Get relative velocity vector in m/s.
        This is a default implementation that should be overridden by subclasses.
        """
        # Get player velocity in body frame (fps)
        player_vel_fps = np.array([
            self.player_sim[prp.u_fps],
            self.player_sim[prp.v_fps],
            self.player_sim[prp.w_fps]
        ])
        
        # Get enemy velocity in body frame (fps)
        enemy_vel_fps = np.array([
            self.enemy_sim[prp.u_fps],
            self.enemy_sim[prp.v_fps],
            self.enemy_sim[prp.w_fps]
        ])
        
        # Convert to m/s and calculate relative velocity
        player_vel_mps = player_vel_fps * 0.3048
        enemy_vel_mps = enemy_vel_fps * 0.3048
        
        # Simple relative velocity calculation (enemy - player)
        relative_vel_mps = enemy_vel_mps - player_vel_mps
        
        return relative_vel_mps
    
    def observe_first_state(self) -> np.ndarray:
        self._new_episode_init()
        state = self.get_state()
        return state
    
    def get_state(self):
        """Get combined state from both aircraft"""
        state = [self.get_prop(prop) for prop in self.state_variables]
        return state
    
    def get_state_norm(self):
        """Get normalized combined state from both aircraft"""
        state = [(np.clip(self.get_prop(prop), prop.min, prop.max) - prop.min) / (prop.max - prop.min) 
                for prop in self.state_variables]
        return state
    
    def get_player_state(self):
        """Get state specifically for player aircraft"""
        return [self.get_player_prop(prop) for prop in self.state_variables]
    
    def get_enemy_state(self):
        """Get state specifically for enemy aircraft"""
        return [self.get_enemy_prop(prop) for prop in self.state_variables]
    
    def get_relative_state(self):
        """Get relative state between player and enemy aircraft"""
        player_pos = self.get_player_position()
        enemy_pos = self.get_enemy_position()
        
        # Calculate relative position, distance, bearing, etc.
        relative_distance = np.linalg.norm(np.array(player_pos) - np.array(enemy_pos))
        relative_bearing = math.atan2(enemy_pos[1] - player_pos[1], enemy_pos[0] - player_pos[0])
        
        return {
            'relative_distance': relative_distance,
            'relative_bearing': relative_bearing,
            'player_position': player_pos,
            'enemy_position': enemy_pos
        }
    
    def get_player_position(self):
        """Get player aircraft position"""
        return (
            self.player_sim[prp.longitude_geoc_deg],
            self.player_sim[prp.latitude_geod_deg],
            self.player_sim[prp.altitude_sl_ft]
        )
    
    def get_enemy_position(self):
        """Get enemy aircraft position"""
        return (
            self.enemy_sim[prp.lat_geod_deg],
            self.enemy_sim[prp.lng_geoc_deg],
            self.enemy_sim[prp.altitude_sl_ft]
        )
    
    @abstractmethod
    def get_props_to_output(self) -> Tuple:
        pass

    def set_sim(self, sim):
        self.player_sim = sim
    
    def set_enemy_sim(self, sim):
        self.enemy_sim = sim
    
    def _new_episode_init(self):
        """Initialize both aircraft for new episode"""
        
        # Setup autopilots
        if self.use_autopilot:
            self.player_autopilot = AutoPilot(self.player_sim)
            assert self.player_sim.sim_frequency_hz >= self.AUTOPILOT_FREQ
            self.player_autopilot_update_interval = self.player_sim.sim_frequency_hz // self.AUTOPILOT_FREQ
        
        # Reset control targets to initial values
        self.player_target_roll = 0.0
        self.player_target_pitch = self.CRUISE_PITCH
        self.player_target_throttle = 0.8
        
        self.current_step = 0

    def get_state_space(self) -> gym.Space:
        state_lows = np.array([state_var.min for state_var in self.state_variables])
        state_highs = np.array([state_var.max for state_var in self.state_variables])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype='float')

    @abstractmethod
    def get_initial_conditions(self) -> Dict[Property, float]:
        pass

    @abstractmethod
    def get_player_initial_conditions(self) -> Dict[Property, float]:
        """Get initial conditions for player aircraft"""
        pass

    @abstractmethod     
    def get_prop(self, prop) -> float:
        """Get property value from combined state (implement based on your needs)"""
        pass
    
    def get_player_prop(self, prop) -> float:
        """Get property value from player aircraft"""
        if type(prop) == BoundedProperty or type(prop) == Property:
            return self.player_sim[prop]
        if prop.name == "h-sl-mt":
            return self.player_sim[prp.altitude_sl_ft] * 0.3048
        if prop.name == "ic/h-sl-mt":
            return self.player_sim[prp.initial_altitude_ft] * 0.3048
        return None
    
    def get_enemy_prop(self, prop) -> float:
        """Get property value from enemy aircraft"""
        if type(prop) == BoundedProperty or type(prop) == Property:
            return self.enemy_sim[prop]
        if prop.name == "h-sl-mt":
            return self.enemy_sim[prp.altitude_sl_ft] * 0.3048
        if prop.name == "ic/h-sl-mt":
            return self.enemy_sim[prp.initial_altitude_ft] * 0.3048
        return None

    @abstractmethod
    def _is_terminal(self, state) -> bool:
        """Check if episode should terminate"""
        for _state in state:
            if math.isnan(_state):
                print("NaN Error")
                return True
        if self.max_steps - self.current_step <= 0:
            return True
        return False

    @abstractmethod
    def calculate_reward(self, state, done) -> float:
        """Calculate reward based on combined state"""
        pass