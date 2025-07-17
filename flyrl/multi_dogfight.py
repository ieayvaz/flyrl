import numpy as np
import math
from typing import Dict, Sequence, Tuple, List
from flyrl.manuevers import TacticalFlightManeuvers
from flyrl.enemy_controller import EnemyWaypointController
from flyrl.properties import BoundedProperty, DerivedProperty, Property
import flyrl.properties as prp
from flyrl.simulation import Simulation
from flyrl.aircraft import Aircraft
from flyrl.multiaircraft_tasks import MultiAircraftFlightTask
from flyrl.utils import mt2ft, ft2mt
from flyrl import geoutils
from flyrl.enemy_controller import EnemyControllerFactory

class MultiAircraftDogfightTask(MultiAircraftFlightTask):
    """Multi-aircraft dogfight task with AI enemy using waypoint navigation"""
    
    INITIAL_HEADING_DEG = 120.0
    GROUND_ALTITUDE_MT = 575
    THROTTLE_CMD = 0.8
    MIXTURE_CMD = 0.8
    
    def __init__(self, step_frequency_hz: float, 
                 player_sim: Simulation, enemy_sim: Simulation,
                 player_aircraft: Aircraft, enemy_aircraft: Aircraft,
                 enemy_difficulty: str = 'hard',
                 max_time_s: float = 30.0, debug: bool = False):
        
        # Define state variables for dogfight - adjusted for smaller area
        distance = DerivedProperty("distance", "Distance between aircraft", 0, 2500) 
        distance_x = DerivedProperty("distance_x", "Distance in x axis", -1000, 1000)  
        distance_y = DerivedProperty("distance_y", "Distance in y axis", -1000, 1000) 
        distance_z = DerivedProperty("distance_z", "Distance in z axis", -250, 250)   
        
        # Angular states as sin/cos components
        enemy_roll_sin = DerivedProperty("enemy_roll_sin", "Enemy roll sine", -1, 1)
        enemy_roll_cos = DerivedProperty("enemy_roll_cos", "Enemy roll cosine", -1, 1)
        enemy_pitch_sin = DerivedProperty("enemy_pitch_sin", "Enemy pitch sine", -1, 1)
        enemy_pitch_cos = DerivedProperty("enemy_pitch_cos", "Enemy pitch cosine", -1, 1)
        enemy_heading_sin = DerivedProperty("enemy_heading_sin", "Enemy heading sine", -1, 1)
        enemy_heading_cos = DerivedProperty("enemy_heading_cos", "Enemy heading cosine", -1, 1)
        
        # Own aircraft angular states
        own_roll_sin = DerivedProperty("own_roll_sin", "Own roll sine", -1, 1)
        own_roll_cos = DerivedProperty("own_roll_cos", "Own roll cosine", -1, 1)
        own_pitch_sin = DerivedProperty("own_pitch_sin", "Own pitch sine", -1, 1)
        own_pitch_cos = DerivedProperty("own_pitch_cos", "Own pitch cosine", -1, 1)
        own_heading_sin = DerivedProperty("own_heading_sin", "Own heading sine", -1, 1)
        own_heading_cos = DerivedProperty("own_heading_cos", "Own heading cosine", -1, 1)
        
        # 3D LOS error as sin/cos components
        los_azimuth_error_sin = DerivedProperty("los_azimuth_error_sin", "LOS azimuth error sine", -1, 1)
        los_azimuth_error_cos = DerivedProperty("los_azimuth_error_cos", "LOS azimuth error cosine", -1, 1)
        los_elevation_error_sin = DerivedProperty("los_elevation_error_sin", "LOS elevation error sine", -1, 1)
        los_elevation_error_cos = DerivedProperty("los_elevation_error_cos", "LOS elevation error cosine", -1, 1)

        relative_velocity_x = DerivedProperty("relative_velocity_x", "Relative velocity X", -50, 50)
        relative_velocity_y = DerivedProperty("relative_velocity_y", "Relative velocity Y", -50, 50)
        relative_velocity_z = DerivedProperty("relative_velocity_z", "Relative velocity Z", -25, 25)
        
        state_variables = (
            distance_x, distance_y, distance_z,
            own_roll_sin, own_roll_cos, own_pitch_sin, own_pitch_cos, own_heading_sin, own_heading_cos,
            enemy_roll_sin, enemy_roll_cos, enemy_pitch_sin, enemy_pitch_cos, enemy_heading_sin, enemy_heading_cos,
            los_azimuth_error_sin, los_azimuth_error_cos, los_elevation_error_sin, los_elevation_error_cos,
            relative_velocity_x, relative_velocity_y, relative_velocity_z,
            prp.p_radps, prp.q_radps, prp.r_radps
        )
        
        super().__init__(
            state_variables=state_variables,
            max_time_s=max_time_s,
            step_frequency_hz=step_frequency_hz,
            player_sim=player_sim,
            enemy_sim=enemy_sim,
            debug=debug,
            use_autopilot=True,
            enemy_use_autopilot=True
        )
        
        self.player_aircraft = player_aircraft
        self.enemy_aircraft = enemy_aircraft
        
        self.min_lock_duration = 4.0
        #self.second_lock_duration = 4.0
        self.good_lock_duration = 2.0
        self.maneuver_controller = TacticalFlightManeuvers(
            max_roll_deg=60.0,
            max_pitch_deg=25.0,
            max_g_limit=6.0,
            cruise_throttle=0.8,
            combat_throttle=0.95,
            attack_throttle=1.0
        )

        # Store property references for easier access
        self.distance_prp = distance
        self.los_azimuth_error_sin_prp = los_azimuth_error_sin
        self.los_azimuth_error_cos_prp = los_azimuth_error_cos
        self.los_elevation_error_sin_prp = los_elevation_error_sin
        self.los_elevation_error_cos_prp = los_elevation_error_cos
        
        # Enemy AI controller
        self.enemy_difficulty = enemy_difficulty
        self.enemy_controller = None
        self.successful_locks = 0
        
        # Visualization
        self.visualization_freq = self.step_frequency_hz / 10 if self.step_frequency_hz > 10 else 1
        
        if self.debug:
            try:
                from flyrl.visualizer_multi import DogfightVisualizer
                self.visualizer = DogfightVisualizer()
            except ImportError:
                print("Warning: DogfightVisualizer not available")
                self.visualizer = None

    def _get_enemy_relative_info(self) -> Tuple[float, float, float]:
        """
        Override base class method with proper geometric calculations for dogfight task.
        """
        # Use the existing methods from this class for accurate calculations
        bearing_deg = self.get_3d_los_azimuth_error()  # This is already relative to aircraft heading
        elevation_deg = self.get_3d_los_elevation_error()  # This is already relative to aircraft pitch
        range_m = self.get_distance()
        
        return bearing_deg, elevation_deg, range_m
    
    def _get_relative_velocity_mps(self) -> np.ndarray:
        """
        Override base class method with proper velocity calculations.
        """
        # Use the existing method from this class
        return self.get_relative_velocity()
    
    def _generate_enemy_action(self) -> Tuple[float, float, float]:
        """Generate enemy AI action using difficulty-appropriate controller"""
        if self.enemy_controller is None:
            # Initialize controller with current position
            enemy_pos = self.get_enemy_position()
            
            # Create controller based on difficulty
            if self.enemy_difficulty.lower() == 'easy':
                self.enemy_controller = EnemyControllerFactory.create_controller(
                    'easy', 
                    enemy_pos,
                    altitude_range=(675, 725),
                    line_length=800.0,  # Length of line to follow
                    speed=0.65  # Slower speed for easy opponent
                )
            elif self.enemy_difficulty.lower() == 'medium':
                self.enemy_controller = EnemyControllerFactory.create_controller(
                    'medium',
                    enemy_pos,
                    altitude_range=(675, 725),
                    turn_probability=0.003,  # Adjust turn frequency
                    turn_angle_range=(20, 60),  # Moderate turn angles
                    straight_time_range=(6.0, 12.0),  # Time between turns
                    speed=0.75
                )
            else:  # hard
                self.enemy_controller = EnemyControllerFactory.create_controller(
                    'hard',
                    enemy_pos,
                    waypoint_radius=250.0,
                    altitude_range=(675, 725),
                    waypoint_threshold=50.0,
                    waypoint_timeout=12.0
                )
        
        # Update enemy controller
        current_pos = self.get_enemy_position()
        dt = 1.0 / self.step_frequency_hz
        
        desired_heading, desired_altitude, desired_throttle = self.enemy_controller.update(current_pos, dt)
        
        return desired_heading, desired_altitude, desired_throttle
    
    def get_player_initial_conditions(self) -> Dict[Property, float]:
        """Get initial conditions for player aircraft"""
        extra_conditions = {
            prp.initial_u_fps: self.player_aircraft.get_cruise_speed_fps(),
            prp.initial_v_fps: 0,
            prp.initial_w_fps: 0,
            prp.initial_p_radps: 0,
            prp.initial_q_radps: 0,
            prp.initial_r_radps: 0,
            prp.initial_roc_fpm: 0,
            prp.initial_heading_deg: self.INITIAL_HEADING_DEG,
        }
        
        conditions = {**self.base_initial_conditions, **extra_conditions}
        # Altitude: 675-725m above sea level (100-150m above ground)
        conditions[prp.initial_altitude_ft] = mt2ft(675 + np.random.random() * 50)
        # Smaller position randomization for 1000x1000m area
        conditions[prp.initial_latitude_geod_deg] = 51.3781 + (np.random.random() - 0.5) * 0.009 * 0.6  # ~500m variation
        conditions[prp.initial_longitude_geoc_deg] = -2.3273 + (np.random.random() - 0.5) * 0.009 * 0.6 # ~500m variation
        
        return conditions
    
    def get_enemy_initial_conditions(self) -> Dict[Property, float]:
        """Get initial conditions for enemy aircraft"""
        extra_conditions = {
            prp.initial_u_fps: self.enemy_aircraft.get_cruise_speed_fps(),
            prp.initial_v_fps: 0,
            prp.initial_w_fps: 0,
            prp.initial_p_radps: 0,
            prp.initial_q_radps: 0,
            prp.initial_r_radps: 0,
            prp.initial_roc_fpm: 0,
            prp.initial_heading_deg: self.INITIAL_HEADING_DEG + 180,  # Start facing opposite direction
        }
        
        conditions = {**self.base_initial_conditions, **extra_conditions}
        # Altitude: 675-725m above sea level (100-150m above ground)
        conditions[prp.initial_altitude_ft] = mt2ft(675 + np.random.random() * 50)
        # Smaller position randomization for 1000x1000m area
        conditions[prp.initial_latitude_geod_deg] = 51.3781 + (np.random.random() - 0.5) * 0.009 * 0.6  # ~500m variation
        conditions[prp.initial_longitude_geoc_deg] = -2.3273 + (np.random.random() - 0.5) * 0.009 * 0.6 # ~500m variation
        
        return conditions
    
    def get_initial_conditions(self) -> Dict[Property, float]:
        """This method is not used in multi-aircraft setup"""
        return self.get_player_initial_conditions()
    
    def _new_episode_init(self):
        """Initialize both aircraft for new episode"""
        super()._new_episode_init()
        self.previous_P = None
        
        # Set throttle and mixture for both aircraft
        self.player_sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)
        self.enemy_sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)
        
        # Reset enemy controller
        self.enemy_controller = None

                # Reset lock tracking
        self.lock_start_time = None
        self.current_lock_duration = 0.0
        self.total_lock_time = 0.0
        self.max_lock_duration = 0.0
        self.lock_count = 0
        self.successful_locks = 0
        self.good_locks = 0
        self.second_locks = 0

            # Add tracking for reward analysis
        self.episode_rewards = []
        self.episode_locks = []
        
        # Set origin for coordinate system
        self.origin = np.array([
            self.player_sim[prp.initial_latitude_geod_deg],
            self.player_sim[prp.initial_longitude_geoc_deg],
            self.player_sim[prp.initial_altitude_ft] * 0.3048
        ])
    
    def get_props_to_output(self) -> Tuple:
        """Properties to output for logging/debugging"""
        return (prp.roll_rad, prp.pitch_rad, prp.heading_deg, prp.altitude_sl_ft)
    
    def get_prop(self, prop) -> float:
        """Get property value from combined state"""
        # Try to get from player aircraft first
        value = self.get_player_prop(prop)
        if value is not None:
            return value
        
        # Handle derived properties
        if prop.name == "distance":
            return float(self.get_distance())
        elif prop.name == "distance_x":
            return float(self.get_distance_v()[0])
        elif prop.name == "distance_y":
            return float(self.get_distance_v()[1])
        elif prop.name == "distance_z":
            return float(self.get_distance_v()[2])
        
        # Own aircraft angular states as sin/cos
        elif prop.name == "own_roll_sin":
            return float(math.sin(self.get_player_prop(prp.roll_rad)))
        elif prop.name == "own_roll_cos":
            return float(math.cos(self.get_player_prop(prp.roll_rad)))
        elif prop.name == "own_pitch_sin":
            return float(math.sin(self.get_player_prop(prp.pitch_rad)))
        elif prop.name == "own_pitch_cos":
            return float(math.cos(self.get_player_prop(prp.pitch_rad)))
        elif prop.name == "own_heading_sin":
            return float(math.sin(math.radians(self.get_player_heading())))
        elif prop.name == "own_heading_cos":
            return float(math.cos(math.radians(self.get_player_heading())))
        
        # Enemy aircraft angular states as sin/cos
        elif prop.name == "enemy_roll_sin":
            return float(math.sin(self.get_enemy_prop(prp.roll_rad)))
        elif prop.name == "enemy_roll_cos":
            return float(math.cos(self.get_enemy_prop(prp.roll_rad)))
        elif prop.name == "enemy_pitch_sin":
            return float(math.sin(self.get_enemy_prop(prp.pitch_rad)))
        elif prop.name == "enemy_pitch_cos":
            return float(math.cos(self.get_enemy_prop(prp.pitch_rad)))
        elif prop.name == "enemy_heading_sin":
            return float(math.sin(math.radians(self.get_enemy_heading())))
        elif prop.name == "enemy_heading_cos":
            return float(math.cos(math.radians(self.get_enemy_heading())))
        
        # 3D LOS error as sin/cos
        elif prop.name == "los_azimuth_error_sin":
            return float(math.sin(math.radians(self.get_3d_los_azimuth_error())))
        elif prop.name == "los_azimuth_error_cos":
            return float(math.cos(math.radians(self.get_3d_los_azimuth_error())))
        elif prop.name == "los_elevation_error_sin":
            return float(math.sin(math.radians(self.get_3d_los_elevation_error())))
        elif prop.name == "los_elevation_error_cos":
            return float(math.cos(math.radians(self.get_3d_los_elevation_error())))
        
                # New relative velocity properties
        if prop.name == "relative_velocity_x":
            return float(self.get_relative_velocity()[0])
        elif prop.name == "relative_velocity_y":
            return float(self.get_relative_velocity()[1])
        elif prop.name == "relative_velocity_z":
            return float(self.get_relative_velocity()[2])
        
        return None
    
    def get_relative_velocity(self) -> np.ndarray:
        """Get relative velocity vector (enemy - player)"""
        # Get player velocity in NED frame
        player_vel_body = np.array([
            self.player_sim[prp.u_fps], 
            self.player_sim[prp.v_fps], 
            self.player_sim[prp.w_fps]
        ])
        R_b2n_player = geoutils.body_to_ned_rotation(
            self.player_sim[prp.roll_rad], 
            self.player_sim[prp.pitch_rad], 
            self.player_sim[prp.heading_deg] * math.pi / 180
        )
        player_vel_ned = R_b2n_player @ player_vel_body
        
        # Get enemy velocity in NED frame
        enemy_vel_body = np.array([
            self.enemy_sim[prp.u_fps], 
            self.enemy_sim[prp.v_fps], 
            self.enemy_sim[prp.w_fps]
        ])
        R_b2n_enemy = geoutils.body_to_ned_rotation(
            self.enemy_sim[prp.roll_rad], 
            self.enemy_sim[prp.pitch_rad], 
            self.enemy_sim[prp.heading_deg] * math.pi / 180
        )
        enemy_vel_ned = R_b2n_enemy @ enemy_vel_body
        
        # Convert to ENU and return relative velocity
        player_vel_enu = np.array([player_vel_ned[1], player_vel_ned[0], -player_vel_ned[2]])
        enemy_vel_enu = np.array([enemy_vel_ned[1], enemy_vel_ned[0], -enemy_vel_ned[2]])
        
        return enemy_vel_enu - player_vel_enu
    
    def get_player_heading(self) -> float:
        """Get player aircraft heading"""
        return self.player_sim[prp.heading_deg]
    
    def get_enemy_heading(self) -> float:
        """Get enemy aircraft heading"""
        return self.enemy_sim[prp.heading_deg]
    
    def get_geo_pos(self) -> np.ndarray:
        """Get player aircraft geographical position"""
        return np.array([
            self.player_sim[prp.lat_geod_deg],
            self.player_sim[prp.lng_geoc_deg],
            self.player_sim[prp.altitude_sl_ft] * 0.3048
        ])
    
    def get_enemy_geo_pos(self) -> np.ndarray:
        """Get enemy aircraft geographical position"""
        return np.array([
            self.enemy_sim[prp.lat_geod_deg],
            self.enemy_sim[prp.lng_geoc_deg],
            self.enemy_sim[prp.altitude_sl_ft] * 0.3048
        ])
    
    def get_pos(self) -> np.ndarray:
        """Get player aircraft position in local coordinates"""
        return geoutils.lla_2_enu(self.get_geo_pos(), self.origin)
    
    def get_enemy_pos(self) -> np.ndarray:
        """Get enemy aircraft position in local coordinates"""
        return geoutils.lla_2_enu(self.get_enemy_geo_pos(), self.origin)
    
    def get_distance(self) -> float:
        """Get distance between aircraft"""
        return np.linalg.norm(self.get_enemy_pos() - self.get_pos())
    
    def get_distance_v(self) -> np.ndarray:
        """Get distance vector from player to enemy"""
        return self.get_enemy_pos() - self.get_pos()
    
    def get_3d_los_azimuth(self) -> float:
        """Get azimuth angle of LOS vector in degrees (0-360)"""
        dist = self.get_distance_v()
        angle_rad = math.atan2(dist[1], dist[0])
        angle_deg = math.degrees(angle_rad)
        return (90 - angle_deg + 360) % 360
    
    def get_3d_los_elevation(self) -> float:
        """Get elevation angle of LOS vector in degrees (-90 to +90)"""
        dist = self.get_distance_v()
        horizontal_dist = math.sqrt(dist[0]**2 + dist[1]**2)
        elevation_rad = math.atan2(dist[2], horizontal_dist)
        return math.degrees(elevation_rad)
    
    def get_3d_los_azimuth_error(self) -> float:
        """Get azimuth error between LOS and aircraft heading"""
        los_azimuth = self.get_3d_los_azimuth()
        heading = self.get_player_heading()
        
        error = los_azimuth - heading
        
        # Normalize to [-180, 180]
        if error > 180:
            error -= 360
        elif error < -180:
            error += 360
        
        return error
    
    def get_3d_los_elevation_error(self) -> float:
        """Get elevation error between LOS and aircraft pitch"""
        los_elevation = self.get_3d_los_elevation()
        pitch = math.degrees(self.get_player_prop(prp.pitch_rad))
        
        error = los_elevation - pitch
        
        # Normalize to [-180, 180]
        if error > 180:
            error -= 360
        elif error < -180:
            error += 360
        
        return error
    
    def is_locked(self) -> bool:
        """Check if player has locked onto enemy using curriculum parameters"""
        
        if self.get_distance() > 50:
            return False
        if abs(self.get_3d_los_azimuth_error()) > 25:
            return False
        if abs(self.get_3d_los_elevation_error()) > 18:
            return False
        return True
    
    def distance_limit(self) -> bool:
        """Check if aircraft are too far apart - adjusted for smaller area"""
        return self.get_distance() > 2500  # Reduced from 2500m
    
    def altitude_limit(self) -> bool:
        """Check if player aircraft is too low"""
        return self.get_player_prop(prp.altitude_sl_mt) < 600
    
    def _is_terminal(self, state) -> bool:
        """Check if episode should terminate"""
        # Check for NaN values
        for _state in state:
            if math.isnan(_state):
                if self.debug:
                    print("NaN Error")
                return True
        
        # Check step limit
        if self.max_steps - self.current_step <= 0:
            return True
        # if self.current_lock_duration >= self.min_lock_duration:
        #     return True
        # Check custom termination conditions
        if self.successful_locks > 0:
            return True
        
        return False
    
    def update_lock_tracking(self):
        """Improved lock duration tracking with better state management"""
        dt = 1.0 / self.step_frequency_hz
        
        if self.is_locked():
            if self.lock_start_time is None:
                # Lock just started
                self.lock_start_time = self.current_step * dt
                self.lock_count += 1
            
            # Update current lock duration
            self.current_lock_duration = (self.current_step * dt) - self.lock_start_time
            self.total_lock_time += dt
            self.max_lock_duration = max(self.max_lock_duration, self.current_lock_duration)
            
            # Check if this becomes a successful lock (only mark once per lock)
            if (self.current_lock_duration >= self.min_lock_duration and 
                self.current_lock_duration - dt < self.min_lock_duration):
                self.successful_locks += 1
            elif(self.current_lock_duration >= self.good_lock_duration and 
                self.current_lock_duration - dt < self.good_lock_duration):
                self.good_locks += 1
                
        else:
            # Not locked - reset current lock tracking
            if self.lock_start_time is not None:
                # Lock was broken
                self.lock_start_time = None
                self.current_lock_duration = 0.0
        
    def calculate_reward(self, state, done) -> float:
        """Simplified reward function for better learning"""
        distance = self.get_distance()
        azimuth_error = abs(self.get_3d_los_azimuth_error())
        elevation_error = abs(self.get_3d_los_elevation_error())
        
        # Your existing rewards
        distance_reward = -distance / 2500.0
        heading_reward = -azimuth_error / 180.0
        elevation_reward = -elevation_error / 90.0
        lock_bonus = 5.0 if self.is_locked() else 0.0
        step_penalty = -0.01
        
        # SIMPLE STABILITY: penalize extreme attitudes AND rates
        roll_penalty = -max(0, abs(self.get_player_prop(prp.roll_rad)) - 0.8) * 5  # 45 degrees
        pitch_penalty = -max(0, abs(self.get_player_prop(prp.pitch_rad)) - 0.35) * 5  # 20 degrees
        
        # Rate penalties (p=roll rate, q=pitch rate)
        roll_rate_penalty = -max(0, abs(self.get_player_prop(prp.p_radps)) - 1.0) * 3  # 1 rad/s
        pitch_rate_penalty = -max(0, abs(self.get_player_prop(prp.q_radps)) - 0.8) * 3  # 0.8 rad/s
        
        step_reward = (distance_reward + 2.0 * heading_reward + elevation_reward + 
                    lock_bonus + step_penalty + roll_penalty + pitch_penalty + 
                    roll_rate_penalty + pitch_rate_penalty)
        
        # Terminal rewards
        terminal_reward = 0.0
        if done:
            if self.successful_locks > 0:
                terminal_reward = 500.0
            elif self.good_locks > 0:
                terminal_reward = 200.0
            elif self.lock_count > 0:
                terminal_reward = 50.0
            elif self.distance_limit():
                terminal_reward = -20.0
            elif self.altitude_limit():
                terminal_reward = -30.0
        
        total_reward = step_reward + terminal_reward
        return total_reward

    
    def get_episode_stats(self) -> dict:
        """Get comprehensive episode statistics for analysis"""
        return {
            'successful_locks': self.successful_locks,
            'total_lock_time': self.total_lock_time,
            'max_lock_duration': self.max_lock_duration,
            'lock_count': self.lock_count,
            'final_distance': self.get_distance(),
            'final_azimuth_error': abs(self.get_3d_los_azimuth_error()),
            'final_elevation_error': abs(self.get_3d_los_elevation_error()),
            'episode_length': self.current_step,
            'total_reward': sum(self.episode_rewards) if hasattr(self, 'episode_rewards') else 0
        }
    
    def get_episode_success(self):
        return self.successful_locks > 0
    
    def task_step(self, action, sim_steps: int):
        """Override task_step to add debug output and visualization"""
        if self.debug:
            out_props = [
                self.distance_prp
            ]
            for _prop in out_props:
                ##print(f"{_prop.name}: {self.get_prop(_prop)}")
                pass
            print("\n")
        # Call parent task_step
        self.update_lock_tracking()
        result = super().task_step(action, sim_steps)
        
        # Update visualization
        if self.debug and hasattr(self, 'visualizer') and self.visualizer:
            if self.current_step % self.visualization_freq == 0:
                self.visualizer.update_from_simulation(self)
                self.visualizer.update_plot()

            # V_body = np.array([self.enemy_sim[prp.u_fps], self.enemy_sim[prp.v_fps], self.enemy_sim[prp.w_fps]])
            # R_b2n = geoutils.body_to_ned_rotation(self.enemy_sim[prp.roll_rad], self.enemy_sim[prp.pitch_rad], self.enemy_sim[prp.heading_deg]* math.pi / 180)
            # V_ned = R_b2n @ V_body
            # enemy_speed = np.linalg.norm(V_ned)
        
        return result
