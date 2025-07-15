from flyrl.PID import PID
from flyrl.simulation import Simulation
import flyrl.properties as prp
import math

class AutoPilot():
    def __init__(self, simulation: Simulation, P : float = 0.1, I : float = 0.01, D : float = 0.0):
        self.sim = simulation
        self.roll_pid = PID(P,I,D)
        self.pitch_pid = PID(P,I,D)
        self.set_target_roll(0)
        self.set_target_pitch(2)

    def set_target_roll(self, target_roll : float):
        self.roll_pid.SetPoint = target_roll

    def set_target_pitch(self, target_pitch : float):
        self.pitch_pid.SetPoint = target_pitch

    def normalize(self, value, min=-1, max=1):
        # if value = 700, and max = 20, return 20
        # if value = -200, and min = -20, return -20
        if (value > max):
            return max
        elif (value < min):
            return min
        else:
            return value
        
    def generate_ail_ctrl(self):
        current_roll = (self.sim[prp.roll_rad]*180.0)/math.pi
        self.roll_pid.update(current_roll)
        new_ail_ctrl = self.roll_pid.output
        new_ail_ctrl = self.normalize(new_ail_ctrl, -1, 1)
        return new_ail_ctrl

    def generate_elv_ctrl(self):
        current_pitch = (self.sim[prp.pitch_rad]*180.0)/math.pi
        self.pitch_pid.update(current_pitch)
        new_elv_ctrl = self.pitch_pid.output
        new_elv_ctrl = self.normalize(new_elv_ctrl, -1, 1)
        return new_elv_ctrl
    
    def generate_controls(self, target_roll : float = 0.0, target_pitch : float = 2.0):
        self.set_target_pitch(target_pitch)
        self.set_target_roll(target_roll)

        return self.generate_ail_ctrl(), -self.generate_elv_ctrl()

    def update_roll(self):
        new_ail_ctrl = self.generate_ail_ctrl()
        self.sim[prp.aileron_cmd] = new_ail_ctrl

    def update_pitch(self):
        new_elv_ctrl = self.generate_elv_ctrl()
        self.sim[prp.elevator_cmd] = new_elv_ctrl

    def update(self, target_roll : float = 0.0, target_pitch : float = 0.2):
        self.set_target_pitch(target_pitch)
        self.set_target_roll(target_roll)

        self.update_roll()
        self.update_pitch()

