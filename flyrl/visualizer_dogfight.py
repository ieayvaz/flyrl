import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import matplotlib.animation as animation
from typing import Dict, List, Tuple, Optional
import math
from flyrl import geoutils
import flyrl.properties as prp


class Arrow3D(FancyArrowPatch):
    """3D arrow for matplotlib"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class DogfightVisualizer:
    """3D visualizer for dogfight scenarios"""
    
    def __init__(self, max_history_points: int = 100, update_interval: int = 100):
        """
        Initialize the 3D visualizer
        
        Args:
            max_history_points: Maximum number of trajectory points to keep
            update_interval: Animation update interval in milliseconds
        """
        self.max_history_points = max_history_points
        self.update_interval = update_interval
        
        # Data storage
        self.own_trajectory = []
        self.enemy_trajectory = []
        self.own_orientation_history = []
        self.enemy_orientation_history = []
        
        # Matplotlib setup
        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Aircraft markers and trajectories
        self.own_aircraft_marker = None
        self.enemy_aircraft_marker = None
        self.own_trajectory_line = None
        self.enemy_trajectory_line = None
        self.own_orientation_arrow = None
        self.enemy_orientation_arrow = None
        
        # Information text
        self.info_text = None
        
        # Engagement zone
        self.engagement_zone = None
        
        # Animation object
        self.animation = None
        
        # Setup the plot
        self._setup_plot()
        
    def _setup_plot(self):
        """Initialize the plot elements"""
        # Set labels and title
        self.ax.set_xlabel('East (m)')
        self.ax.set_ylabel('North (m)')
        self.ax.set_zlabel('Altitude (m)')
        self.ax.set_title('Dogfight 3D Visualization')
        
        # Set initial view
        self.ax.view_init(elev=20, azim=45)
        
        # Initialize empty trajectory lines
        self.own_trajectory_line, = self.ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7, label='Own Aircraft')
        self.enemy_trajectory_line, = self.ax.plot([], [], [], 'r-', linewidth=2, alpha=0.7, label='Enemy Aircraft')
        
        # Initialize aircraft markers
        self.own_aircraft_marker, = self.ax.plot([], [], [], 'bo', markersize=8, label='Own Position')
        self.enemy_aircraft_marker, = self.ax.plot([], [], [], 'rs', markersize=8, label='Enemy Position')
        
        # Add legend
        self.ax.legend(loc='upper right')
        
        # Add info text
        self.info_text = self.fig.text(0.02, 0.98, '', transform=self.fig.transFigure, 
                                      verticalalignment='top', fontsize=10, 
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Set grid
        self.ax.grid(True, alpha=0.3)
        
    def update_positions(self, own_pos: Tuple[float, float, float], 
                        enemy_pos: Tuple[float, float, float],
                        own_orientation: Tuple[float, float, float] = None,
                        enemy_orientation: Tuple[float, float, float] = None):
        """
        Update aircraft positions and orientations
        
        Args:
            own_pos: (x, y, z) position of own aircraft in ENU coordinates
            enemy_pos: (x, y, z) position of enemy aircraft in ENU coordinates
            own_orientation: (roll, pitch, heading) of own aircraft in radians
            enemy_orientation: (roll, pitch, heading) of enemy aircraft in radians
        """
        # Add positions to trajectory history
        self.own_trajectory.append(own_pos)
        self.enemy_trajectory.append(enemy_pos)
        
        # Add orientations to history
        if own_orientation:
            self.own_orientation_history.append(own_orientation)
        if enemy_orientation:
            self.enemy_orientation_history.append(enemy_orientation)
        
        # Limit trajectory length
        if len(self.own_trajectory) > self.max_history_points:
            self.own_trajectory.pop(0)
        if len(self.enemy_trajectory) > self.max_history_points:
            self.enemy_trajectory.pop(0)
        if len(self.own_orientation_history) > self.max_history_points:
            self.own_orientation_history.pop(0)
        if len(self.enemy_orientation_history) > self.max_history_points:
            self.enemy_orientation_history.pop(0)
            
    def update_from_simulation(self, env):
        """
        Update positions directly from simulation and target objects
        
        Args:
            sim: Simulation object
            target: Target object
            origin: Origin point for ENU conversion [lat, lon, alt]
        """
        # Get own aircraft position
        target = env.get_target()
        origin = env.origin
        own_lat = env.get_prop(prp.lat_geod_deg)
        own_lon = env.get_prop(prp.lng_geoc_deg)
        own_alt = env.get_prop(prp.altitude_sl_ft) * 0.3048  # Convert to meters
        own_geo_pos = np.array([own_lat, own_lon, own_alt])
        own_enu_pos = geoutils.lla_2_enu(own_geo_pos, origin)
        
        # Get enemy aircraft position
        enemy_lat = target["Lat"]
        enemy_lon = target["Lon"]
        enemy_alt = target["Alt"]
        enemy_geo_pos = np.array([enemy_lat, enemy_lon, enemy_alt])
        enemy_enu_pos = geoutils.lla_2_enu(enemy_geo_pos, origin)
        
        # Get orientations
        own_orientation = (env.get_prop(prp.roll_rad), env.get_prop(prp.pitch_rad), env.get_prop(prp.heading_deg) * math.pi / 180)
        enemy_orientation = (target["Roll"], target["Pitch"], target["Heading"] * math.pi / 180)
        # Update positions
        self.update_positions(tuple(own_enu_pos), tuple(enemy_enu_pos), 
                            own_orientation, enemy_orientation)
        
    def _get_aircraft_orientation_vector(self, orientation: Tuple[float, float, float],
                                    length: float = 50.0) -> np.ndarray:
        """
        Calculate aircraft nose direction vector from orientation

        Args:
            orientation: (roll, pitch, heading) in radians
            length: Length of the orientation vector

        Returns:
            3D vector pointing in aircraft's forward direction
        """
        roll, pitch, heading = orientation
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)
        cos_p = math.cos(pitch)
        sin_p = math.sin(pitch)

        forward = np.array([
            cos_p * sin_h,  # East component
            cos_p * cos_h,  # North component
            sin_p           # Up component
        ]) * length

        return forward
        
    def _calculate_engagement_metrics(self) -> Dict:
        """Calculate engagement metrics for display"""
        if not self.own_trajectory or not self.enemy_trajectory:
            return {}
            
        own_pos = np.array(self.own_trajectory[-1])
        enemy_pos = np.array(self.enemy_trajectory[-1])
        
        # Distance
        distance = np.linalg.norm(enemy_pos - own_pos)
        
        # Relative position
        rel_pos = enemy_pos - own_pos
        
        # Altitude difference
        alt_diff = enemy_pos[2] - own_pos[2]
        
        metrics = {
            'distance': distance,
            'altitude_diff': alt_diff,
            'relative_bearing': math.atan2(rel_pos[1], rel_pos[0]) * 180 / math.pi,
            'num_points': len(self.own_trajectory)
        }
        
        return metrics
        
    def update_plot(self):
        """Update the 3D plot with current data"""
        if not self.own_trajectory or not self.enemy_trajectory:
            return
            
        # Update trajectory lines
        if len(self.own_trajectory) > 1:
            own_traj = np.array(self.own_trajectory)
            self.own_trajectory_line.set_data_3d(own_traj[:, 0], own_traj[:, 1], own_traj[:, 2])
            
        if len(self.enemy_trajectory) > 1:
            enemy_traj = np.array(self.enemy_trajectory)
            self.enemy_trajectory_line.set_data_3d(enemy_traj[:, 0], enemy_traj[:, 1], enemy_traj[:, 2])
        
        # Update aircraft markers
        own_pos = self.own_trajectory[-1]
        enemy_pos = self.enemy_trajectory[-1]
        
        self.own_aircraft_marker.set_data_3d([own_pos[0]], [own_pos[1]], [own_pos[2]])
        self.enemy_aircraft_marker.set_data_3d([enemy_pos[0]], [enemy_pos[1]], [enemy_pos[2]])
        
        # Update orientation arrows
        if self.own_orientation_history and self.enemy_orientation_history:
            self._update_orientation_arrows()
        
        # Update plot limits
        self._update_plot_limits()
        
        # Update info text
        metrics = self._calculate_engagement_metrics()
        if metrics:
            info_str = f"Distance: {metrics['distance']:.1f}m\n"
            info_str += f"Alt Diff: {metrics['altitude_diff']:.1f}m\n"
            info_str += f"Rel Bearing: {metrics['relative_bearing']:.1f}°\n"
            info_str += f"Points: {metrics['num_points']}"
            self.info_text.set_text(info_str)
        
        # Redraw
        plt.pause(0.00001)
        self.fig.canvas.draw()
        
    def _update_orientation_arrows(self):
        """Update aircraft orientation arrows"""
        if not (self.own_orientation_history and self.enemy_orientation_history):
            return
            
        own_pos = np.array(self.own_trajectory[-1])
        enemy_pos = np.array(self.enemy_trajectory[-1])
        
        # Remove old arrows
        if self.own_orientation_arrow:
            self.own_orientation_arrow.remove()
        if self.enemy_orientation_arrow:
            self.enemy_orientation_arrow.remove()
            
        # Calculate orientation vectors
        own_forward = self._get_aircraft_orientation_vector(self.own_orientation_history[-1])
        enemy_forward = self._get_aircraft_orientation_vector(self.enemy_orientation_history[-1])
        
        # Create new arrows
        self.own_orientation_arrow = Arrow3D([own_pos[0], own_pos[0] + own_forward[0]],
                                           [own_pos[1], own_pos[1] + own_forward[1]],
                                           [own_pos[2], own_pos[2] + own_forward[2]],
                                           mutation_scale=20, lw=2, arrowstyle="-|>", color="blue")
        
        self.enemy_orientation_arrow = Arrow3D([enemy_pos[0], enemy_pos[0] + enemy_forward[0]],
                                             [enemy_pos[1], enemy_pos[1] + enemy_forward[1]],
                                             [enemy_pos[2], enemy_pos[2] + enemy_forward[2]],
                                             mutation_scale=20, lw=2, arrowstyle="-|>", color="red")
        
        self.ax.add_artist(self.own_orientation_arrow)
        self.ax.add_artist(self.enemy_orientation_arrow)
        
    def _update_plot_limits(self):
        """Update plot limits based on current data"""
        if not self.own_trajectory or not self.enemy_trajectory:
            return
            
        all_positions = np.array(self.own_trajectory + self.enemy_trajectory)
        
        # Add some padding
        padding = 100  # meters
        
        x_min, x_max = all_positions[:, 0].min() - padding, all_positions[:, 0].max() + padding
        y_min, y_max = all_positions[:, 1].min() - padding, all_positions[:, 1].max() + padding
        z_min, z_max = all_positions[:, 2].min() - padding, all_positions[:, 2].max() + padding
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_zlim(z_min, z_max)
        
    def add_engagement_zone(self, center: Tuple[float, float, float], radius: float):
        """Add a spherical engagement zone visualization"""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        self.engagement_zone = self.ax.plot_wireframe(x, y, z, alpha=0.2, color='gray')
        
    def start_animation(self, update_function=None):
        """Start real-time animation"""
        def animate(frame):
            if update_function:
                update_function()
            self.update_plot()
            return []
            
        self.animation = animation.FuncAnimation(self.fig, animate, interval=self.update_interval, 
                                               blit=False, repeat=True)
        
    def stop_animation(self):
        """Stop the animation"""
        if self.animation:
            self.animation.event_source.stop()
            
    def save_animation(self, filename: str, duration: int = 10):
        """Save animation as gif or mp4"""
        if self.animation:
            self.animation.save(filename, writer='pillow' if filename.endswith('.gif') else 'ffmpeg',
                              fps=1000//self.update_interval)
            
    def clear_trajectories(self):
        """Clear trajectory history"""
        self.own_trajectory.clear()
        self.enemy_trajectory.clear()
        self.own_orientation_history.clear()
        self.enemy_orientation_history.clear()
        
    def show(self):
        """Display the plot"""
        plt.show()
        
    def close(self):
        """Close the plot"""
        plt.close(self.fig)


# Example usage function
def example_usage():
    """Example of how to use the visualizer with the DogfightTask"""
    
    # Create visualizer
    visualizer = DogfightVisualizer(max_history_points=200, update_interval=50)
    
    # Example simulation loop integration
    def update_visualization(dogfight_task):
        """Update visualization from dogfight task"""
        # Update kinematics cache
        dogfight_task._update_kinematics_cache()
        
        # Update visualizer
        visualizer.update_from_simulation(
            dogfight_task.sim, 
            dogfight_task.target, 
            dogfight_task.origin
        )
        
        # Update the plot
        visualizer.update_plot()
    
    return visualizer, update_visualization


if __name__ == "__main__":
    # Simple test with dummy data
    visualizer = DogfightVisualizer()
    
    # Add some test trajectory points
    for i in range(50):
        t = i * 0.1
        own_pos = (100 * np.cos(t), 100 * np.sin(t), 1000 + 10 * t)
        enemy_pos = (50 * np.cos(t + np.pi/4), 50 * np.sin(t + np.pi/4), 1100 + 5 * t)
        
        own_orientation = (0, 0.1 * np.sin(t), t)
        enemy_orientation = (0, -0.1 * np.sin(t), t + np.pi/4)
        
        visualizer.update_positions(own_pos, enemy_pos, own_orientation, enemy_orientation)
    
    # Update and show
    visualizer.update_plot()
    visualizer.show()