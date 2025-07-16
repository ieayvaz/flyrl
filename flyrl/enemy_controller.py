import numpy as np
import math
from typing import Dict, Sequence, Tuple, List
from flyrl.utils import mt2ft, ft2mt
from flyrl import geoutils
import random


class EnemyLineFollowerController:
    """Easy AI controller that follows a straight line"""
    
    def __init__(self, initial_position: Tuple[float, float, float], 
                 altitude_range: Tuple[float, float] = (625, 725),
                 line_length: float = 2000.0,
                 speed: float = 0.7):
        """
        Initialize line follower controller
        
        Args:
            initial_position: (lat, lon, alt) starting position
            altitude_range: (min_alt, max_alt) altitude range in meters above sea level
            line_length: Length of the line to follow in meters
            speed: Throttle setting (0.5-1.0)
        """
        self.initial_position = initial_position
        self.altitude_range = altitude_range
        self.line_length = line_length
        self.speed = speed
        
        # Generate a random line direction
        self.line_heading = random.uniform(0, 360)
        self.desired_altitude = random.uniform(altitude_range[0], altitude_range[1])
        
        # Calculate end point of the line
        self.line_start = initial_position
        self.line_end = self._calculate_end_point()
        
        # Direction flag (1 for forward, -1 for backward)
        self.direction = 1
    
    def _calculate_end_point(self) -> Tuple[float, float, float]:
        """Calculate the end point of the line"""
        # Convert line heading to radians
        heading_rad = math.radians(self.line_heading)
        
        # Calculate lat/lon offset
        lat_offset = (self.line_length * math.cos(heading_rad)) / 111320.0
        lon_offset = (self.line_length * math.sin(heading_rad)) / (111320.0 * math.cos(math.radians(self.initial_position[0])))
        
        end_lat = self.initial_position[0] + lat_offset
        end_lon = self.initial_position[1] + lon_offset
        
        return (end_lat, end_lon, self.desired_altitude)
    
    def update(self, current_position: Tuple[float, float, float], dt: float) -> Tuple[float, float, float]:
        """
        Update controller and return desired heading, altitude, and throttle
        
        Args:
            current_position: (lat, lon, alt) current aircraft position
            dt: time step in seconds
            
        Returns:
            (desired_heading, desired_altitude, desired_throttle)
        """
        # Check if we've reached either end of the line
        dist_to_end = self._calculate_distance(current_position, self.line_end)
        dist_to_start = self._calculate_distance(current_position, self.line_start)
        
        # If we're close to an end, reverse direction
        if dist_to_end < 50 and self.direction == 1:
            self.direction = -1
        elif dist_to_start < 50 and self.direction == -1:
            self.direction = 1
        
        # Determine target based on direction
        if self.direction == 1:
            target = self.line_end
        else:
            target = self.line_start
        
        # Calculate desired heading to target
        desired_heading = self._calculate_bearing(current_position, target)
        
        return desired_heading, self.desired_altitude, self.speed
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Calculate distance between two positions"""
        lat1, lon1, alt1 = pos1
        lat2, lon2, alt2 = pos2
        
        # Haversine formula for great circle distance
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in meters
        earth_radius = 6371000
        horizontal_distance = earth_radius * c
        
        # Include altitude difference
        alt_diff = alt2 - alt1
        return math.sqrt(horizontal_distance**2 + alt_diff**2)
    
    def _calculate_bearing(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Calculate bearing from pos1 to pos2 in degrees"""
        lat1, lon1, _ = pos1
        lat2, lon2, _ = pos2
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360 degrees
        return (bearing_deg + 360) % 360


class EnemyOccasionalTurnController:
    """Medium difficulty AI controller that flies mostly straight but occasionally turns"""
    
    def __init__(self, initial_position: Tuple[float, float, float], 
                 altitude_range: Tuple[float, float] = (625, 725),
                 turn_probability: float = 0.1,  # Probability per update step
                 turn_angle_range: Tuple[float, float] = (30, 90),
                 straight_time_range: Tuple[float, float] = (8.0, 15.0),
                 speed: float = 0.75):
        """
        Initialize occasional turn controller
        
        Args:
            initial_position: (lat, lon, alt) starting position
            altitude_range: (min_alt, max_alt) altitude range in meters above sea level
            turn_probability: Probability of initiating a turn per update step
            turn_angle_range: (min_angle, max_angle) range of turn angles in degrees
            straight_time_range: (min_time, max_time) time to fly straight after turn
            speed: Throttle setting (0.5-1.0)
        """
        self.initial_position = initial_position
        self.altitude_range = altitude_range
        self.turn_probability = turn_probability
        self.turn_angle_range = turn_angle_range
        self.straight_time_range = straight_time_range
        self.speed = speed
        
        # Current state
        self.current_heading = random.uniform(0, 360)
        self.desired_altitude = random.uniform(altitude_range[0], altitude_range[1])
        self.time_since_last_turn = 0.0
        self.min_straight_time = random.uniform(straight_time_range[0], straight_time_range[1])
        
        # Track if we're in a turn
        self.is_turning = False
        self.turn_start_time = 0.0
        self.turn_duration = 0.0
        self.target_heading = self.current_heading
    
    def update(self, current_position: Tuple[float, float, float], dt: float) -> Tuple[float, float, float]:
        """
        Update controller and return desired heading, altitude, and throttle
        
        Args:
            current_position: (lat, lon, alt) current aircraft position
            dt: time step in seconds
            
        Returns:
            (desired_heading, desired_altitude, desired_throttle)
        """
        self.time_since_last_turn += dt
        
        # Check if we should initiate a turn
        if (not self.is_turning and 
            self.time_since_last_turn > self.min_straight_time and
            random.random() < self.turn_probability):
            
            self._initiate_turn()
        
        # Handle ongoing turn
        if self.is_turning:
            self.turn_start_time += dt
            if self.turn_start_time >= self.turn_duration:
                self._complete_turn()
            else:
                # Gradually turn towards target heading
                progress = self.turn_start_time / self.turn_duration
                self.current_heading = self._interpolate_heading(
                    self.current_heading, self.target_heading, progress
                )
        
        # Occasionally change altitude slightly
        if random.random() < 0.001:  # Very low probability
            self.desired_altitude = random.uniform(self.altitude_range[0], self.altitude_range[1])
        
        return self.current_heading, self.desired_altitude, self.speed
    
    def _initiate_turn(self):
        """Initiate a new turn"""
        self.is_turning = True
        self.turn_start_time = 0.0
        self.turn_duration = random.uniform(3.0, 6.0)  # Turn takes 3-6 seconds
        
        # Calculate new heading
        turn_angle = random.uniform(self.turn_angle_range[0], self.turn_angle_range[1])
        if random.random() < 0.5:
            turn_angle = -turn_angle  # Turn left or right
        
        self.target_heading = (self.current_heading + turn_angle) % 360
        self.time_since_last_turn = 0.0
    
    def _complete_turn(self):
        """Complete the current turn"""
        self.is_turning = False
        self.current_heading = self.target_heading
        self.min_straight_time = random.uniform(self.straight_time_range[0], self.straight_time_range[1])
    
    def _interpolate_heading(self, start_heading: float, end_heading: float, progress: float) -> float:
        """Interpolate between two headings, handling 0/360 wrap-around"""
        # Handle wrap-around
        diff = end_heading - start_heading
        if diff > 180:
            diff -= 360
        elif diff < -180:
            diff += 360
        
        result = start_heading + diff * progress
        return result % 360


class EnemyWaypointController:
    """Hard AI controller that generates random waypoints for enemy aircraft (your original)"""
    
    def __init__(self, initial_position: Tuple[float, float, float], 
                 waypoint_radius: float = 150.0,
                 altitude_range: Tuple[float, float] = (625, 725),
                 waypoint_threshold: float = 75.0,
                 waypoint_timeout: float = 15.0):
        """
        Initialize waypoint controller
        
        Args:
            initial_position: (lat, lon, alt) starting position
            waypoint_radius: Maximum distance from initial position for waypoints
            altitude_range: (min_alt, max_alt) altitude range in meters above sea level
            waypoint_threshold: Distance threshold to consider waypoint reached
            waypoint_timeout: Time before generating new waypoint
        """
        self.initial_position = initial_position
        self.waypoint_radius = waypoint_radius
        self.altitude_range = altitude_range
        self.waypoint_threshold = waypoint_threshold
        self.waypoint_timeout = waypoint_timeout
        
        self.current_waypoint = None
        self.waypoint_history = []
        self.time_since_last_waypoint = 0.0
        
        # Generate initial waypoint
        self.generate_new_waypoint()
    
    def generate_new_waypoint(self):
        """Generate a new random waypoint"""
        # Generate random position within radius
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(30, self.waypoint_radius)
        
        # Convert to lat/lon offset
        lat_offset = (distance * math.cos(angle)) / 111320.0
        lon_offset = (distance * math.sin(angle)) / (111320.0 * math.cos(math.radians(self.initial_position[0])))
        
        waypoint_lat = self.initial_position[0] + lat_offset
        waypoint_lon = self.initial_position[1] + lon_offset
        waypoint_alt = random.uniform(self.altitude_range[0], self.altitude_range[1])
        
        self.current_waypoint = (waypoint_lat, waypoint_lon, waypoint_alt)
        self.waypoint_history.append(self.current_waypoint)
        self.time_since_last_waypoint = 0.0
        
        # Keep only last 5 waypoints in history
        if len(self.waypoint_history) > 5:
            self.waypoint_history.pop(0)
    
    def update(self, current_position: Tuple[float, float, float], dt: float) -> Tuple[float, float, float]:
        """
        Update controller and return desired heading, altitude, and throttle
        
        Args:
            current_position: (lat, lon, alt) current aircraft position
            dt: time step in seconds
            
        Returns:
            (desired_heading, desired_altitude, desired_throttle)
        """
        self.time_since_last_waypoint += dt
        
        # Check if we've reached the waypoint or timed out
        distance_to_waypoint = self._calculate_distance(current_position, self.current_waypoint)
        
        if distance_to_waypoint < self.waypoint_threshold or self.time_since_last_waypoint > self.waypoint_timeout:
            self.generate_new_waypoint()
            distance_to_waypoint = self._calculate_distance(current_position, self.current_waypoint)
        
        # Calculate desired heading to waypoint
        desired_heading = self._calculate_bearing(current_position, self.current_waypoint)
        
        # Desired altitude is the waypoint altitude
        desired_altitude = self.current_waypoint[2]
        
        # Calculate desired throttle based on distance
        if distance_to_waypoint < 100:
            desired_throttle = 0.6
        elif distance_to_waypoint < 200:
            desired_throttle = 0.7
        else:
            desired_throttle = 0.8
        
        desired_throttle = np.clip(desired_throttle, 0.5, 1.0)
        
        return desired_heading, desired_altitude, desired_throttle
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Calculate distance between two positions"""
        lat1, lon1, alt1 = pos1
        lat2, lon2, alt2 = pos2
        
        # Haversine formula for great circle distance
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        c = 2 * math.asin(math.sqrt(a))
        
        # Earth radius in meters
        earth_radius = 6371000
        horizontal_distance = earth_radius * c
        
        # Include altitude difference
        alt_diff = alt2 - alt1
        return math.sqrt(horizontal_distance**2 + alt_diff**2)
    
    def _calculate_bearing(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Calculate bearing from pos1 to pos2 in degrees"""
        lat1, lon1, _ = pos1
        lat2, lon2, _ = pos2
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360 degrees
        return (bearing_deg + 360) % 360


class EnemyControllerFactory:
    """Factory class to create different difficulty level controllers"""
    
    @staticmethod
    def create_controller(difficulty: str, initial_position: Tuple[float, float, float], **kwargs):
        """
        Create an enemy controller based on difficulty level
        
        Args:
            difficulty: 'easy', 'medium', or 'hard'
            initial_position: (lat, lon, alt) starting position
            **kwargs: Additional parameters for specific controllers
            
        Returns:
            Enemy controller instance
        """
        if difficulty.lower() == 'easy':
            return EnemyLineFollowerController(initial_position, **kwargs)
        elif difficulty.lower() == 'medium':
            return EnemyOccasionalTurnController(initial_position, **kwargs)
        elif difficulty.lower() == 'hard':
            return EnemyWaypointController(initial_position, **kwargs)
        else:
            raise ValueError(f"Unknown difficulty level: {difficulty}. Use 'easy', 'medium', or 'hard'.")