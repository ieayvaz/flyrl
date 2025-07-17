import gymnasium as gym
import enum
from flyrl.tasks import Task, HeadingControlTask, TurnHeadingControlTask
from flyrl.waypoint_travel import WaypointTask
from flyrl.dogfight import DogfightTask
from flyrl.multi_dogfight import MultiAircraftDogfightTask
from flyrl.dogfight_ap import APDogfightTask
from flyrl.aircraft import Aircraft, cessna172P
from flyrl import utils
from flyrl.aircraft import cessna172P, rascal

"""
This script registers all combinations of task, aircraft, shaping settings
 etc. with OpenAI Gym so that they can be instantiated with a gym.make(id)
 command.

The gym_jsbsim.Envs enum stores all registered environments as members with
 their gym id string as value. This allows convenient autocompletion and value
 safety. To use do:
       env = gym.make(gym_jsbsim.Envs.desired_environment.value)
"""

for env_id, (task, plane, shaping, enable_flightgear) in utils.get_env_id_kwargs_map().items():
    if enable_flightgear:
        entry_point = 'flyrl.environment:JsbSimEnv'
    else:
        entry_point = 'flyrl.environment:NoFGJsbSimEnv'
    kwargs = dict(task_type=task,
                  aircraft=plane,
                  shaping=shaping)
    gym.envs.registration.register(id=env_id,
                                   entry_point=entry_point,
                                   kwargs=kwargs)
    
gym.envs.registration.register(id="WaypointC172p",entry_point='flyrl.basic_environment:BasicJsbSimEnv',kwargs=dict(task_type=WaypointTask,aircraft=cessna172P))
gym.envs.registration.register(id="WaypointC172p-debug",entry_point='flyrl.basic_environment:BasicJsbSimEnv',kwargs=dict(task_type=WaypointTask,aircraft=cessna172P,debug=True))
gym.envs.registration.register(id="DogfightC172p-debug",entry_point='flyrl.basic_environment:BasicJsbSimEnv',kwargs=dict(task_type=DogfightTask,aircraft=cessna172P,debug=True))
gym.envs.registration.register(id="DogfightRascal-debug",entry_point='flyrl.basic_environment:BasicJsbSimEnv',kwargs=dict(task_type=DogfightTask,aircraft=rascal,debug=True))
gym.envs.registration.register(id="DogfightRascal",entry_point='flyrl.basic_environment:BasicJsbSimEnv',kwargs=dict(task_type=DogfightTask,aircraft=rascal,debug=False))
#gym.envs.registration.register(id="DogfightAP",entry_point='flyrl.ap_environment:APEnv',kwargs=dict(task_type=DogfightAPTask,aircraft=None,agent_interaction_freq=1,debug=False))
#gym.envs.registration.register(id="DogfightAP-debug",entry_point='flyrl.ap_environment:APEnv',kwargs=dict(task_type=DogfightAPTask,aircraft=None,agent_interaction_freq=1,debug=True))
gym.envs.registration.register(id="DogfightAP2P",entry_point='flyrl.ap_environment:APEnv',kwargs=dict(task_type=APDogfightTask,aircraft=None,agent_interaction_freq=4,debug=False))
gym.envs.registration.register(id="DogfightAP2P-debug",entry_point='flyrl.ap_environment:APEnv',kwargs=dict(task_type=APDogfightTask,aircraft=None,agent_interaction_freq=4,debug=True))
# make an Enum storing every Gym-JSBSim environment ID for convenience and value safety
gym.envs.registration.register(id="MultiDogfightRascal-debug",entry_point='flyrl.multi_env:MultiAircraftJsbSimEnv',kwargs=dict(task_type=MultiAircraftDogfightTask,player_aircraft=rascal,enemy_aircraft=rascal,debug=True))
gym.envs.registration.register(id="MultiDogfightRascal",entry_point='flyrl.multi_env:MultiAircraftJsbSimEnv',kwargs=dict(task_type=MultiAircraftDogfightTask,player_aircraft=rascal,enemy_aircraft=rascal,debug=False))
Envs = enum.Enum.__call__('Envs', [(utils.AttributeFormatter.translate(env_id), env_id)
                                   for env_id in utils.get_env_id_kwargs_map().keys()])
