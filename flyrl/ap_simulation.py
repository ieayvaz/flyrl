import time
from typing import Dict, Union
import flyrl.properties as prp
from dronekit import connect, VehicleMode

# We need a new property to represent the velocity vector
prp.velocity_ned_mps = prp.Property('velocities/ned-velocity-mps', 'Velocity vector in NED frame [m/s]')

class AP_Simulation(object):
    """
    A class which wraps an instance of ArduPilot and manages communication with it.
    MODIFIED to correctly fetch the velocity vector.
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