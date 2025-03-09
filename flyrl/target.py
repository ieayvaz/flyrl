import pandas as pd
import numpy as np

class Target:
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.time = self.data['Time'].iloc[0]  # Start at the first timestamp
    
    def step(self, dts):
        self.time += dts  # Update internal time

    def reset(self):
        self.time = self.data['Time'].iloc[0]
    
    def __getitem__(self, key):
        if key not in self.data.columns:
            raise KeyError(f"State '{key}' not found in data")
        
        # Get surrounding time indices
        t_prev_idx = np.searchsorted(self.data['Time'].values, self.time, side='right') - 1
        t_next_idx = min(t_prev_idx + 1, len(self.data) - 1)
        
        t_prev, t_next = self.data['Time'].iloc[t_prev_idx], self.data['Time'].iloc[t_next_idx]
        value_prev, value_next = self.data[key].iloc[t_prev_idx], self.data[key].iloc[t_next_idx]
        
        # Linear interpolation
        if t_next == t_prev:
            return value_prev
        return value_prev + (value_next - value_prev) * (self.time - t_prev) / (t_next - t_prev)


    