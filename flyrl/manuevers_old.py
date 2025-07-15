import numpy as np
import math
from enum import Enum

class TacticalManeuver(Enum):
    """
    Simplified enumeration of tactical flight maneuvers for air-to-air combat.
    Focused on essential maneuvers that are easy for RL models to learn.
    """
    # Basic maneuver
    STRAIGHT_AND_LEVEL = 0
    
    # Core pursuit maneuvers - these are the bread and butter
    PURE_PURSUIT = 1          # Point directly at enemy
    LEAD_PURSUIT = 2          # Lead the enemy to intercept
    LAG_PURSUIT = 3           # Follow behind enemy's turn
    
    # Vertical lead pursuit - lead pursuit with altitude advantage
    VERTICAL_LEAD_PURSUIT = 4 # Lead pursuit while climbing to 50m above enemy

class TacticalFlightManeuvers:
    """
    Simplified tactical flight maneuvers focused on core combat maneuvers.
    Designed to be easy for RL models to learn and use effectively.
    """

    def __init__(self,
                 max_roll_deg: float = 60.0,
                 max_pitch_deg: float = 25.0,
                 max_g_limit: float = 6.0,
                 cruise_throttle: float = 0.8,
                 combat_throttle: float = 0.95,
                 attack_throttle: float = 1.0,
                 vertical_lead_altitude_m: float = 50.0):
        """
        Initialize tactical maneuvers with aircraft limitations.
        
        Args:
            max_roll_deg: Maximum roll angle in degrees
            max_pitch_deg: Maximum pitch angle in degrees
            max_g_limit: Maximum G-force limit
            cruise_throttle: Normal flight throttle
            combat_throttle: Combat maneuvering throttle
            attack_throttle: Maximum attack throttle
            vertical_lead_altitude_m: Target altitude advantage for vertical lead pursuit
        """
        self.max_roll = max_roll_deg
        self.max_pitch = max_pitch_deg
        self.max_g_limit = max_g_limit
        self.cruise_throttle = cruise_throttle
        self.combat_throttle = combat_throttle
        self.attack_throttle = attack_throttle
        self.vertical_lead_altitude = vertical_lead_altitude_m
        
        # Maneuver-specific parameters
        self.pursuit_roll_gain = 5.0  # Gain for pursuit maneuvers
        self.lead_pursuit_gain = 1.2  # Lead pursuit anticipation factor
        self.vertical_climb_gain = 0.8  # Gain for vertical maneuvering

    def get_maneuver_controls(self,
                              maneuver: TacticalManeuver,
                              current_roll_deg: float,
                              current_pitch_deg: float,
                              enemy_bearing_deg: float,
                              enemy_elevation_deg: float,
                              enemy_range_m: float,
                              relative_velocity_mps: np.ndarray,
                              current_heading_deg: float,
                              current_speed_mps: float,
                              current_altitude_m: float = 0.0,
                              enemy_altitude_m: float = 0.0) -> tuple[float, float, float]:
        """
        Generate tactical maneuver controls based on enemy position and relative geometry.
        
        Args:
            maneuver: Selected tactical maneuver
            current_roll_deg: Current aircraft roll angle
            current_pitch_deg: Current aircraft pitch angle
            enemy_bearing_deg: Bearing to enemy (-180 to 180)
            enemy_elevation_deg: Elevation to enemy (-90 to 90)
            enemy_range_m: Range to enemy in meters
            relative_velocity_mps: Relative velocity vector [x, y, z]
            current_heading_deg: Current aircraft heading
            current_speed_mps: Current aircraft speed
            current_altitude_m: Current UAV altitude
            enemy_altitude_m: Enemy UAV altitude
            
        Returns:
            Tuple of (target_roll_deg, target_pitch_deg, target_throttle)
        """
        target_roll = 0.0
        target_pitch = 0.0
        target_throttle = self.cruise_throttle
        
        # Calculate lead angle for pursuit maneuvers
        if enemy_range_m > 0:
            relative_speed = np.linalg.norm(relative_velocity_mps)
            time_to_intercept = enemy_range_m / max(relative_speed, 1.0)
            lead_angle = min(30.0, time_to_intercept * 10.0)  # Simplified lead calculation
        else:
            lead_angle = 0.0
        
        # Calculate altitude difference for vertical maneuvers
        altitude_diff = current_altitude_m - enemy_altitude_m
        
        if maneuver == TacticalManeuver.STRAIGHT_AND_LEVEL:
            target_roll = 0.0
            target_pitch = 0.0
            target_throttle = self.cruise_throttle
            
        elif maneuver == TacticalManeuver.PURE_PURSUIT:
            # Point directly at enemy
            target_roll = np.clip(enemy_bearing_deg * self.pursuit_roll_gain, -self.max_roll, self.max_roll)
            target_pitch = np.clip(enemy_elevation_deg * 0.8, -self.max_pitch, self.max_pitch)
            target_throttle = self.combat_throttle
            
        elif maneuver == TacticalManeuver.LEAD_PURSUIT:
            # Lead the enemy based on relative motion
            lead_bearing = enemy_bearing_deg + (lead_angle if enemy_bearing_deg > 0 else -lead_angle)
            target_roll = np.clip(lead_bearing * self.pursuit_roll_gain, -self.max_roll, self.max_roll)
            target_pitch = np.clip(enemy_elevation_deg * 0.8, -self.max_pitch, self.max_pitch)
            target_throttle = self.combat_throttle
            
        elif maneuver == TacticalManeuver.LAG_PURSUIT:
            # Follow behind enemy's turn to maintain position
            lag_bearing = enemy_bearing_deg * 0.7  # Reduce bearing angle
            target_roll = np.clip(lag_bearing * self.pursuit_roll_gain, -self.max_roll, self.max_roll)
            target_pitch = np.clip(enemy_elevation_deg * 0.5, -self.max_pitch, self.max_pitch)
            target_throttle = self.cruise_throttle
            
        elif maneuver == TacticalManeuver.VERTICAL_LEAD_PURSUIT:
            # Lead pursuit while climbing to gain altitude advantage
            lead_bearing = enemy_bearing_deg + (lead_angle if enemy_bearing_deg > 0 else -lead_angle)
            target_roll = np.clip(lead_bearing * self.pursuit_roll_gain, -self.max_roll, self.max_roll)
            
            # Calculate desired altitude advantage
            altitude_error = (enemy_altitude_m + self.vertical_lead_altitude) - current_altitude_m
            
            # Combine elevation tracking with altitude advantage seeking
            if altitude_error > 10:  # Need to climb significantly
                # Prioritize climb while maintaining some enemy tracking
                target_pitch = np.clip(self.max_pitch * 0.7, -self.max_pitch, self.max_pitch)
            elif altitude_error < -10:  # Too high, need to descend
                # Gentle descent while maintaining pursuit
                target_pitch = np.clip(-self.max_pitch * 0.3, -self.max_pitch, self.max_pitch)
            else:
                # Close to desired altitude, focus on enemy elevation
                target_pitch = np.clip(enemy_elevation_deg * 0.6, -self.max_pitch, self.max_pitch)
            
            target_throttle = self.combat_throttle
        
        # Apply aircraft limitations
        target_roll = np.clip(target_roll, -self.max_roll, self.max_roll)
        target_pitch = np.clip(target_pitch, -self.max_pitch, self.max_pitch)
        target_throttle = np.clip(target_throttle, 0.0, 1.0)
        
        return target_roll, target_pitch, target_throttle

    def get_maneuver_description(self, maneuver: TacticalManeuver) -> str:
        """Get human-readable description of maneuver"""
        descriptions = {
            TacticalManeuver.STRAIGHT_AND_LEVEL: "Maintain level flight",
            TacticalManeuver.PURE_PURSUIT: "Point directly at enemy",
            TacticalManeuver.LEAD_PURSUIT: "Lead enemy to intercept",
            TacticalManeuver.LAG_PURSUIT: "Follow behind enemy turn",
            TacticalManeuver.VERTICAL_LEAD_PURSUIT: "Lead pursuit with altitude advantage"
        }
        return descriptions.get(maneuver, "Unknown maneuver")

    def get_recommended_maneuver(self, 
                                enemy_bearing_deg: float,
                                enemy_elevation_deg: float,
                                enemy_range_m: float,
                                relative_velocity_mps: np.ndarray,
                                altitude_advantage: float,
                                current_speed_mps: float,
                                current_altitude_m: float = 0.0,
                                enemy_altitude_m: float = 0.0,
                                min_combat_speed: float = 150.0) -> TacticalManeuver:
        """
        Recommend a maneuver based on tactical situation.
        
        Args:
            enemy_bearing_deg: Bearing to enemy
            enemy_elevation_deg: Elevation to enemy
            enemy_range_m: Range to enemy
            relative_velocity_mps: Relative velocity vector
            altitude_advantage: Altitude difference (positive = we're higher)
            current_speed_mps: Current aircraft speed
            current_altitude_m: Current UAV altitude
            enemy_altitude_m: Enemy UAV altitude
            min_combat_speed: Minimum speed for effective combat
            
        Returns:
            Recommended tactical maneuver
        """
        # Check if we need energy (speed or altitude)
        if current_speed_mps < min_combat_speed or altitude_advantage < -100:
            return TacticalManeuver.CLIMB_FOR_ENERGY
        
        # Calculate if we should use vertical lead pursuit
        altitude_diff = current_altitude_m - enemy_altitude_m
        should_use_vertical_lead = (
            enemy_range_m > 200 and  # Medium to long range
            altitude_diff < self.vertical_lead_altitude * 0.8 and  # Not already at good altitude advantage
            abs(enemy_bearing_deg) < 60  # Enemy is not too far to the side
        )
        
        # Close range combat - use appropriate pursuit
        if enemy_range_m < 300:
            if should_use_vertical_lead and altitude_diff < 30:
                return TacticalManeuver.VERTICAL_LEAD_PURSUIT
            elif abs(enemy_bearing_deg) < 15:  # Enemy roughly ahead
                return TacticalManeuver.PURE_PURSUIT
            elif abs(enemy_bearing_deg) < 45:  # Enemy at moderate angle
                return TacticalManeuver.LEAD_PURSUIT
            else:  # Enemy at high angle - use lag to avoid overshoot
                return TacticalManeuver.LAG_PURSUIT
        
        # Medium to long range - position for attack
        else:
            if should_use_vertical_lead:
                return TacticalManeuver.VERTICAL_LEAD_PURSUIT
            elif abs(enemy_bearing_deg) < 30:  # Enemy roughly ahead
                return TacticalManeuver.LEAD_PURSUIT
            else:  # Need to turn toward enemy
                return TacticalManeuver.PURE_PURSUIT
        
        return TacticalManeuver.STRAIGHT_AND_LEVEL

    def get_maneuver_effectiveness(self, 
                                   maneuver: TacticalManeuver,
                                   enemy_range_m: float,
                                   altitude_advantage: float,
                                   bearing_angle: float) -> float:
        """
        Calculate effectiveness score for a maneuver in current situation.
        Useful for RL reward shaping.
        
        Args:
            maneuver: The maneuver to evaluate
            enemy_range_m: Range to enemy
            altitude_advantage: Altitude difference
            bearing_angle: Absolute bearing angle to enemy
            
        Returns:
            Effectiveness score (0.0 to 1.0, higher is better)
        """
        effectiveness = 0.5  # Base effectiveness
        
        if maneuver == TacticalManeuver.VERTICAL_LEAD_PURSUIT:
            # More effective at medium range with good positioning
            if 200 < enemy_range_m < 600:
                effectiveness += 0.3
            # More effective when we don't have altitude advantage
            if altitude_advantage < 30:
                effectiveness += 0.2
            # More effective when enemy is roughly ahead
            if bearing_angle < 45:
                effectiveness += 0.2
        
        elif maneuver == TacticalManeuver.PURE_PURSUIT:
            # More effective at close range
            if enemy_range_m < 300:
                effectiveness += 0.3
            # More effective when enemy is ahead
            if bearing_angle < 30:
                effectiveness += 0.2
                
        elif maneuver == TacticalManeuver.LEAD_PURSUIT:
            # More effective at medium range
            if 300 < enemy_range_m < 800:
                effectiveness += 0.3
            # More effective with some bearing angle
            if 15 < bearing_angle < 60:
                effectiveness += 0.2
        
        return np.clip(effectiveness, 0.0, 1.0)