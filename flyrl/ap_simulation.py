import time
from typing import Dict, Union
import flyrl.properties as prp
from flyrl.aircraft import Aircraft, cessna172P
from dronekit import connect, VehicleMode

class AP_Simulation(object):
    """
    A class which wraps an instance of ArduPilot and manages communication with it.
    """
    ROLL_CHANNEL_MAX=1900
    ROLL_CHANNEL_MIN=1100
    PITCH_CHANNEL_MAX=1900
    PITCH_CHANNEL_MIN=1100
    THROTTLE_CHANNEL_MAX=2000
    THROTTLE_CHANNEL_MIN=1000


    def __init__(self,
                 address : str = '127.0.0.1:14550', controlled=True):
        self.address = address
        self.vehicle = connect(address, wait_ready=True)
        #Need vehicle to be set up, armed, and in air.
        #checks and setting up code can be implemented later
        #but this is not the goal of this class.
        #goal of this class is purely controlling an airplane in the air.
        print(f"CONNECTED TO DEVICE AT {address}")
        self.sim_frequency_hz = 10
        if controlled:
            self.vehicle.mode = VehicleMode("FBWA")

    def __getitem__(self, prop: Union[prp.BoundedProperty, prp.Property]) -> float:
        '''
        Gets a property class and returns the value of property from remote vehicle
        using dronekit.
        '''
        # Mapping between property names and dronekit attributes
        property_map = {
            'h-sl-mt': lambda vehicle: vehicle.location.global_frame.alt,
            'position/h-sl-ft': lambda vehicle: vehicle.location.global_frame.alt * 3.28084,  # Convert meters to feet
            'attitude/pitch-rad': lambda vehicle: vehicle.attitude.pitch,
            'attitude/roll-rad': lambda vehicle: vehicle.attitude.roll,
            'attitude/psi-deg': lambda vehicle: vehicle.heading,
            'position/lat-geod-deg': lambda vehicle: vehicle.location.global_frame.lat,
            'position/long-gc-deg': lambda vehicle: vehicle.location.global_frame.lon,
            'velocities/u-fps': lambda vehicle: vehicle.velocity,
            'fcs/aileron-cmd-norm' : lambda vehicle: self.roll_value,
            'fcs/elevator-cmd-norm' : lambda vehicle: self.pitch_value,
            'fcs/throttle-cmd-norm' : lambda vehicle: self.throttle_value,
        }
        
        # Get the property name
        prop_name = prop.name
        
        # Check if the property is supported
        if prop_name not in property_map:
            raise ValueError(f"Property '{prop_name}' is not supported")
        
        # Get the value using the appropriate mapping function
        value = property_map[prop_name](self.vehicle)
        
        return value

    def __setitem__(self, prop: Union[prp.BoundedProperty, prp.Property], value) -> None:
        if prop.name == 'fcs/aileron-cmd-norm':
            self.roll_value = value
        if prop.name == 'fcs/elevator-cmd-norm':
            self.pitch_value = value
        if prop.name == 'fcs/throttle-cmd-norm':
            self.throttle_value = value

    def control_plane(self, roll_value = None, pitch_value = None, throttle_value = None):
        roll_val = roll_value
        pitch_val = pitch_value
        throttle_val = throttle_value

        if roll_value == None:
            roll_val = self.roll_value
        if pitch_value == None:
            pitch_val = self.pitch_value
        if throttle_value == None:
            throttle_val = self.throttle_value

        #Assuming aileron elevator control values are between -1,1 and throttle is 0,1
        roll_val = int((roll_val + 1)*(self.ROLL_CHANNEL_MAX - self.ROLL_CHANNEL_MIN)/2.0 + self.ROLL_CHANNEL_MIN)
        pitch_val = int((pitch_val + 1)*(self.PITCH_CHANNEL_MAX - self.PITCH_CHANNEL_MIN)/2.0 + self.PITCH_CHANNEL_MIN)
        throttle_val = int((throttle_val)*(self.THROTTLE_CHANNEL_MAX - self.THROTTLE_CHANNEL_MIN) + self.THROTTLE_CHANNEL_MIN)

        self.vehicle.channels.overrides = {
            '1': roll_val,
            '2': pitch_val, 
            '3': throttle_val, 
        }

    def enable_flightgear_output(self):
        return None
    
    def set_simulation_time_factor(self,factor):
        return None

    def run(self) -> bool:
        self.control_plane()

    def close(self):
        self.vehicle.close()
        print(f"CONNECTION CLOSED AT {self.address}")

    def start_engines(self):
        pass

    def set_throttle_mixture_controls(self, throttle_cmd: float, mixture_cmd: float):
        pass

    def set_throttle(self, throttle_cmd: float):
        self.throttle_value = throttle_cmd

    def raise_landing_gear(self):
        pass
