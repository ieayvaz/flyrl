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

class MultiAircraftFlightTask(Task, ABC):
    MAXIMUM_ROLL = 60.0
    MAXIMUM_PITCH = 25.0
    INITIAL_ALTITUDE_FT = 2250
    AUTOPILOT_FREQ = 60.0
    CRUISE_PITCH = 2.0
    ROLL_INCREMENT = 15.0
    PITCH_INCREMENT = 5.0
    THROTTLE_INCREMENT = 0.1
    
    # Action space limits - changed to heading/altitude/throttle
    HEADING_MIN = 0.0      # degrees
    HEADING_MAX = 360.0    # degrees
    ALTITUDE_MIN = 650.0   # meters
    ALTITUDE_MAX = 750.0  # meters
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
                 player_sim: Simulation, enemy_sim: Simulation, 
                 debug: bool = False,
                 action_variables: Tuple = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd, prp.throttle_cmd), 
                 use_autopilot=True,
                 enemy_use_autopilot=True):
        self.debug = debug
        self.state_variables = state_variables
        self.action_variables = action_variables
        self.max_steps = math.ceil(max_time_s * step_frequency_hz)
        
        # Separate simulations for player and enemy
        self.player_sim = player_sim
        self.enemy_sim = enemy_sim
        
        self.step_frequency_hz = step_frequency_hz
        self.use_autopilot = use_autopilot
        self.enemy_use_autopilot = enemy_use_autopilot
        
        self.current_step = 0
        
        # Player aircraft control targets - changed to heading/altitude/throttle
        self.player_target_roll = 0
        self.player_target_pitch = 2.0
        self.player_target_throttle = 1.0
        
        # Enemy aircraft targets (controlled by AI with high-level commands)
        self.enemy_target_heading = 0.0
        self.enemy_target_altitude = self.INITIAL_ALTITUDE_FT * 0.3048  # Convert to meters
        self.enemy_target_throttle = 0.8

    def task_step(self, action, sim_steps: int) -> Tuple[np.ndarray, float, bool, Dict]:
        # Process player action (heading, altitude, throttle)
        self._process_player_action(action)
        
        # Generate enemy action (AI behavior)
        enemy_action = self._generate_enemy_action()
        
        # Apply controls to both aircraft
        self._apply_player_controls()
        self._apply_enemy_controls()
        
        # Run both simulations
        self._run_simulations(sim_steps, action, enemy_action)
        
        # Get combined state
        state = self.get_state()
        done = self._is_terminal(state)
        
        reward = self.calculate_reward(
            dict(zip([prop.name for prop in self.state_variables], state)), 
            done
        )
        
        state_norm = self.get_state_norm()
        self.current_step += 1
        
        return state_norm, reward, done, {}
    
    def _process_player_action(self, action):
            # Process roll (action[0])
        if action[0] == 0:
            self.player_target_roll += self.ROLL_INCREMENT
        elif action[0] == 2:
            self.player_target_roll -= self.ROLL_INCREMENT
        # If action[0] == 1, do nothing (keep roll same)

        # Process pitch (action[1])
        if action[1] == 0:
            self.player_target_pitch += self.PITCH_INCREMENT
        elif action[1] == 2:
            self.player_target_pitch -= self.PITCH_INCREMENT
        # If action[1] == 1, do nothing (keep pitch same)

        # Process throttle (action[2])
        if action[2] == 0:
            self.player_target_throttle += self.THROTTLE_INCREMENT
        elif action[2] == 2:
            self.player_target_throttle -= self.THROTTLE_INCREMENT
        # If action[2] == 1, do nothing (keep throttle same)
    
    def _calculate_control_targets(self) -> Tuple[float, float]:
        """
        Calculate roll and pitch targets based on desired heading and altitude.
        """
        # Get current aircraft state
        current_heading = self.player_sim[prp.heading_deg]
        current_altitude = self.player_sim[prp.altitude_sl_ft] * 0.3048  # Convert to meters
        
        # Calculate heading error
        heading_error = self.player_target_heading - current_heading
        
        # Normalize heading error to [-180, 180]
        while heading_error > 180:
            heading_error -= 360
        while heading_error < -180:
            heading_error += 360
        
        # Calculate desired roll based on heading error
        # Use proportional control with some limits
        roll_gain = 1.0  # Adjust this gain as needed
        target_roll = np.clip(heading_error * roll_gain, -self.MAXIMUM_ROLL, self.MAXIMUM_ROLL)
        
        # Calculate altitude error
        altitude_error = self.player_target_altitude - current_altitude
        
        # Calculate desired pitch based on altitude error
        # Use proportional control with some limits
        pitch_gain = 0.5  # Adjust this gain as needed
        target_pitch = np.clip(altitude_error * pitch_gain, -self.MAXIMUM_PITCH, self.MAXIMUM_PITCH)
        
        return target_roll, target_pitch
    
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
    
    def _process_enemy_action(self, action):
        """Process the enemy's action - high-level commands (heading, altitude, throttle)"""
        desired_heading, desired_altitude, desired_throttle = action
        
        # Set enemy targets directly from AI commands
        self.enemy_target_heading = desired_heading
        self.enemy_target_altitude = desired_altitude
        self.enemy_target_throttle = np.clip(desired_throttle, 0.5, 1.0)
    
    def _apply_player_controls(self):
        """Apply controls to player aircraft using autopilot with heading/altitude commands"""
        if self.use_autopilot:
            # Calculate roll and pitch targets based on desired heading and altitude
            target_roll = self.player_target_roll
            target_pitch = self.player_target_pitch
            
            # Generate control surface commands via autopilot
            _action = self.player_autopilot.generate_controls(target_roll, target_pitch)
        else:
            # Direct control - you would implement direct heading/altitude control here
            _action = (0, 0)

        # Apply control surface commands
        for prop, command in zip((prp.aileron_cmd, prp.elevator_cmd), _action):
            self.player_sim[prop] = command
        
        # Apply throttle directly
        self.player_sim[prp.throttle_cmd] = self.player_target_throttle
    
    def _apply_enemy_controls(self):
        """Apply controls to enemy aircraft using autopilot for heading and altitude"""
        if self.enemy_use_autopilot:
            # Calculate roll and pitch commands to achieve desired heading and altitude
            current_heading = self.enemy_sim[prp.heading_deg]
            current_altitude = self.enemy_sim[prp.altitude_sl_ft] * 0.3048
            
            # Calculate heading error
            heading_error = self.enemy_target_heading - current_heading
            if heading_error > 180:
                heading_error -= 360
            elif heading_error < -180:
                heading_error += 360
            
            # Calculate desired roll based on heading error
            desired_roll = np.clip(heading_error * 1.0, -self.MAXIMUM_ROLL, self.MAXIMUM_ROLL)
            
            # Calculate altitude error and desired pitch
            altitude_error = self.enemy_target_altitude - current_altitude
            desired_pitch = np.clip(altitude_error * 0.5, -self.MAXIMUM_PITCH, self.MAXIMUM_PITCH)
            
            # Use autopilot to generate control commands
            _action = self.enemy_autopilot.generate_controls(desired_roll, desired_pitch)
        else:
            _action = (0, 0)

        for prop, command in zip((prp.aileron_cmd, prp.elevator_cmd), _action):
            self.enemy_sim[prop] = command
        self.enemy_sim[prp.throttle_cmd] = self.enemy_target_throttle
    
    def _run_simulations(self, sim_steps, player_action, enemy_action):
        """Run both simulations synchronously"""
        if self.use_autopilot:
            for i in range(sim_steps):
                if i % self.player_autopilot_update_interval == 0:
                    _action = self.player_autopilot.generate_controls(self.player_target_roll, self.player_target_pitch)
                    for prop, command in zip((prp.aileron_cmd, prp.elevator_cmd), _action):
                        self.player_sim[prop] = command
                    self.player_sim[prp.throttle_cmd] = self.player_target_throttle
                self.player_sim.run()
        else:
            for _ in range(sim_steps):
                self.player_sim.run()
        
        if self.enemy_use_autopilot:
            for i in range(sim_steps):
                self._process_enemy_action(enemy_action)
                if i % self.enemy_autopilot_update_interval == 0:
                    # Recalculate desired roll and pitch based on current state
                    current_heading = self.enemy_sim[prp.heading_deg]
                    current_altitude = self.enemy_sim[prp.altitude_sl_ft] * 0.3048
                    
                    heading_error = self.enemy_target_heading - current_heading
                    if heading_error > 180:
                        heading_error -= 360
                    elif heading_error < -180:
                        heading_error += 360
                    
                    desired_roll = np.clip(heading_error * 1.0, -self.MAXIMUM_ROLL, self.MAXIMUM_ROLL)
                    altitude_error = self.enemy_target_altitude - current_altitude
                    desired_pitch = np.clip(altitude_error * 0.5, -self.MAXIMUM_PITCH, self.MAXIMUM_PITCH)
                    _action = self.enemy_autopilot.generate_controls(desired_roll, desired_pitch)
                    for prop, command in zip((prp.aileron_cmd, prp.elevator_cmd), _action):
                        self.enemy_sim[prop] = command
                self.enemy_sim.run()
        else:
            for _ in range(sim_steps):
                self.enemy_sim.run()
    
    @abstractmethod
    def _generate_enemy_action(self) -> Tuple[float, float, float]:
        """Generate enemy AI action as (desired_heading, desired_altitude, desired_throttle)."""
        pass
    
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
    
    def _new_episode_init(self):
        """Initialize both aircraft for new episode"""
        # Initialize player aircraft
        self.player_sim.start_engines()
        self.player_sim.raise_landing_gear()
        
        # Initialize enemy aircraft
        self.enemy_sim.start_engines()
        self.enemy_sim.raise_landing_gear()
        
        # Setup autopilots
        if self.use_autopilot:
            self.player_autopilot = AutoPilot(self.player_sim)
            assert self.player_sim.sim_frequency_hz >= self.AUTOPILOT_FREQ
            self.player_autopilot_update_interval = self.player_sim.sim_frequency_hz // self.AUTOPILOT_FREQ
        
        if self.enemy_use_autopilot:
            self.enemy_autopilot = AutoPilot(self.enemy_sim)
            assert self.enemy_sim.sim_frequency_hz >= self.AUTOPILOT_FREQ
            self.enemy_autopilot_update_interval = self.enemy_sim.sim_frequency_hz // self.AUTOPILOT_FREQ
        
        # Reset control targets to initial values
        self.player_target_heading = self.player_sim[prp.heading_deg]  # Start with current heading
        self.player_target_altitude = self.player_sim[prp.altitude_sl_ft] * 0.3048  # Convert to meters
        self.player_target_throttle = 0.8
        
        self.current_step = 0

    def get_state_space(self) -> gym.Space:
        state_lows = np.array([state_var.min for state_var in self.state_variables])
        state_highs = np.array([state_var.max for state_var in self.state_variables])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype='float')

    def get_action_space(self) -> gym.Space:
        gym.spaces.MultiDiscrete([3,3,3])

    @abstractmethod
    def get_initial_conditions(self) -> Dict[Property, float]:
        pass

    @abstractmethod
    def get_player_initial_conditions(self) -> Dict[Property, float]:
        """Get initial conditions for player aircraft"""
        pass
    
    @abstractmethod
    def get_enemy_initial_conditions(self) -> Dict[Property, float]:
        """Get initial conditions for enemy aircraft"""
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