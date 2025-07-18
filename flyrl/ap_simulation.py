import time
import math
from typing import Dict, Union, Optional
import flyrl.properties as prp
from dronekit import connect, VehicleMode

# We need a new property to represent the velocity vector
prp.velocity_ned_mps = prp.Property('velocities/ned-velocity-mps', 'Velocity vector in NED frame [m/s]')

class AP_Simulation(object):
    """
    A class which wraps an instance of ArduPilot and manages communication with it.
    MODIFIED to correctly fetch the velocity vector and angular rates.
    """
    ROLL_CHANNEL_MAX=1900
    ROLL_CHANNEL_MIN=1100
    PITCH_CHANNEL_MAX=1900
    PITCH_CHANNEL_MIN=1100
    THROTTLE_CHANNEL_MAX=1600
    THROTTLE_CHANNEL_MIN=1300

    def __init__(self, address : str = '127.0.0.1:14550', controlled=True):
        self.address = address
        self.vehicle = connect(address, wait_ready=True)
        # Need vehicle to be set up, armed, and in air.
        print(f"CONNECTED TO DEVICE AT {address}")
        self.sim_frequency_hz = 120 # This is a placeholder, real timing is handled by the task loop sleep
        if controlled:
            self.vehicle.mode = VehicleMode("FBWA")
            self.roll_value = 0.0
            self.pitch_value = 0.2
            self.throttle_value = 0.8
        
        # Initialize angular rate calculation variables
        self.prev_attitude: Optional[Dict[str, float]] = None
        self.prev_time: Optional[float] = None
        self.angular_rates: Dict[str, float] = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
        self.rate_filter_alpha = 0.3  # Low-pass filter coefficient for smoothing rates

    def __getitem__(self, prop: Union[prp.BoundedProperty, prp.Property]) -> float:
        '''
        Gets a property class and returns the value of property from remote vehicle
        using dronekit.
        '''
        # Mapping between property names and dronekit attributes
        property_map = {
            'h-sl-mt': lambda vehicle: vehicle.location.global_frame.alt,
            'position/h-sl-ft': lambda vehicle: vehicle.location.global_frame.alt * 3.28084,
            'attitude/pitch-rad': lambda vehicle: vehicle.attitude.pitch,
            'attitude/roll-rad': lambda vehicle: vehicle.attitude.roll,
            'attitude/psi-deg': lambda vehicle: vehicle.heading,
            'position/lat-geod-deg': lambda vehicle: vehicle.location.global_frame.lat,
            'position/long-gc-deg': lambda vehicle: vehicle.location.global_frame.lon,
            # NEW: Correctly named property for the velocity vector [North, East, Down] in m/s
            'velocities/ned-velocity-mps': lambda vehicle: self.vehicle.velocity,
            'fcs/aileron-cmd-norm' : lambda vehicle: self.roll_value,
            'fcs/elevator-cmd-norm' : lambda vehicle: self.pitch_value,
            'fcs/throttle-cmd-norm' : lambda vehicle: self.throttle_value,
            'velocities/u-fps' : lambda vehicle: self.vehicle.velocity[0],
            'velocities/v-fps' : lambda vehicle: self.vehicle.velocity[1],
            'velocities/w-fps' : lambda vehicle: self.vehicle.velocity[2],
            # NEW: Angular rates in rad/s (computed from attitude derivatives)
            'velocities/p-rad_sec': lambda vehicle: self.get_roll_rate(),
            'velocities/q-rad_sec': lambda vehicle: self.get_pitch_rate(),
            'velocities/r-rad_sec': lambda vehicle: self.get_yaw_rate(),
        }
        
        prop_name = prop.name
        
        if prop_name not in property_map:
            raise ValueError(f"Property '{prop_name}' is not supported in AP_Simulation")
        
        value = property_map[prop_name](self.vehicle)
        return value

    def __setitem__(self, prop: Union[prp.BoundedProperty, prp.Property], value) -> None:
        if prop.name == 'fcs/aileron-cmd-norm':
            self.roll_value = value
        elif prop.name == 'fcs/elevator-cmd-norm':
            self.pitch_value = value
        elif prop.name == 'fcs/throttle-cmd-norm':
            self.throttle_value = value

    def control_plane(self, roll_value = None, pitch_value = None, throttle_value = None):
        roll_val = self.roll_value if roll_value is None else roll_value
        pitch_val = self.pitch_value if pitch_value is None else pitch_value
        throttle_val = self.throttle_value if throttle_value is None else throttle_value

        # Assuming aileron/elevator values are between -1,1 and throttle is 0,1
        roll_val = int((roll_val + 1)*(self.ROLL_CHANNEL_MAX - self.ROLL_CHANNEL_MIN)/2.0 + self.ROLL_CHANNEL_MIN)
        pitch_val = int((-pitch_val + 1)*(self.PITCH_CHANNEL_MAX - self.PITCH_CHANNEL_MIN)/2.0 + self.PITCH_CHANNEL_MIN) # Pitch is inverted
        throttle_val = int(throttle_val * (self.THROTTLE_CHANNEL_MAX - self.THROTTLE_CHANNEL_MIN) + self.THROTTLE_CHANNEL_MIN)

        self.vehicle.channels.overrides = {
            '1': roll_val,
            '2': pitch_val, 
            '3': throttle_val, 
        }

    def run(self) -> bool:
        self.control_plane()

    def close(self):
        # Clear overrides before closing
        if self.vehicle.mode == "FBWA":
            self.vehicle.channels.overrides = {}
        self.vehicle.close()
        print(f"CONNECTION CLOSED AT {self.address}")

    def set_throttle(self, throttle_cmd: float):
        self.throttle_value = throttle_cmd

    def _update_angular_rates(self):
        """
        Update angular rates by differentiating attitude angles.
        Uses a low-pass filter to smooth the computed rates.
        """
        current_time = time.time()
        current_attitude = {
            'roll': self.vehicle.attitude.roll,
            'pitch': self.vehicle.attitude.pitch,
            'yaw': self.vehicle.heading * math.pi / 180.0  # Convert to radians
        }
        
        if self.prev_attitude is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            
            if dt > 0:  # Avoid division by zero
                # Calculate raw rates
                roll_rate = self._angle_diff(current_attitude['roll'], self.prev_attitude['roll']) / dt
                pitch_rate = self._angle_diff(current_attitude['pitch'], self.prev_attitude['pitch']) / dt
                yaw_rate = self._angle_diff(current_attitude['yaw'], self.prev_attitude['yaw']) / dt
                
                # Apply low-pass filter for smoothing
                alpha = self.rate_filter_alpha
                self.angular_rates['roll'] = alpha * roll_rate + (1 - alpha) * self.angular_rates['roll']
                self.angular_rates['pitch'] = alpha * pitch_rate + (1 - alpha) * self.angular_rates['pitch']
                self.angular_rates['yaw'] = alpha * yaw_rate + (1 - alpha) * self.angular_rates['yaw']
        
        self.prev_attitude = current_attitude
        self.prev_time = current_time

    def _angle_diff(self, angle1: float, angle2: float) -> float:
        """
        Calculate the difference between two angles, handling wrap-around.
        Returns the shortest angular distance between the angles.
        """
        diff = angle1 - angle2
        # Normalize to [-pi, pi]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def get_angular_rates(self) -> Dict[str, float]:
        """
        Convenience method to get all angular rates at once.
        Returns a dictionary with roll, pitch, and yaw rates in rad/s.
        """
        self._update_angular_rates()
        return {
            'roll_rate': self.angular_rates['roll'],
            'pitch_rate': self.angular_rates['pitch'],
            'yaw_rate': self.angular_rates['yaw']
        }

    def get_roll_rate(self) -> float:
        """Get roll rate in rad/s"""
        self._update_angular_rates()
        return self.angular_rates['roll']

    def get_pitch_rate(self) -> float:
        """Get pitch rate in rad/s"""
        self._update_angular_rates()
        return self.angular_rates['pitch']

    def get_yaw_rate(self) -> float:
        """Get yaw rate in rad/s"""
        self._update_angular_rates()
        return self.angular_rates['yaw']