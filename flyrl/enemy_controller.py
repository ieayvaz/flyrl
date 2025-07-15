import numpy as np
import math
from typing import Dict, Sequence, Tuple, List
from flyrl.utils import mt2ft, ft2mt
from flyrl import geoutils
import random


class EnemyWaypointController:
    """AI controller that generates random waypoints for enemy aircraft"""
    
    def __init__(self, initial_position: Tuple[float, float, float], 
                 waypoint_radius: float = 150.0,  # Reduced from 1000m to 400m for smaller area
                 altitude_range: Tuple[float, float] = (625, 725),  # 50-150m above 575m ground
                 waypoint_threshold: float = 75.0):  # Reduced from 200m to 75m
        """
        Initialize waypoint controller
        
        Args:
            initial_position: (lat, lon, alt) starting position
            waypoint_radius: Maximum distance from initial position for waypoints
            altitude_range: (min_alt, max_alt) altitude range in meters above sea level
            waypoint_threshold: Distance threshold to consider waypoint reached
        """
        self.initial_position = initial_position
        self.waypoint_radius = waypoint_radius
        self.altitude_range = altitude_range
        self.waypoint_threshold = waypoint_threshold
        
        self.current_waypoint = None
        self.waypoint_history = []
        self.time_since_last_waypoint = 0.0
        self.waypoint_timeout = 15.0 
        
        # Generate initial waypoint
        self.generate_new_waypoint()
    
    def generate_new_waypoint(self):
        """Generate a new random waypoint"""
        # Generate random position within radius
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(30, self.waypoint_radius)  # Minimum distance of 100m
        
        # Convert to lat/lon offset (approximate)
        lat_offset = (distance * math.cos(angle)) / 111320.0  # meters to degrees
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
        
        # Calculate desired throttle based on distance (adjusted for smaller distances)
        if distance_to_waypoint < 100:  # Reduced from 300m
            desired_throttle = 0.6  # Slower when close
        elif distance_to_waypoint < 200:  # Reduced from 600m
            desired_throttle = 0.7  # Medium speed
        else:
            desired_throttle = 0.8  # Faster when fa
 
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