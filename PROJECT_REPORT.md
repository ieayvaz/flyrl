# FlyRL Framework: Project Report

## 1. Introduction/Overview

`flyrl` is a Python framework designed for reinforcement learning (RL) research in the context of flight simulation. Its primary purpose is to provide a robust platform for developing and testing RL agents capable of performing various flight tasks. The framework integrates the **JSBSim** open-source flight dynamics model as its core physics engine. It utilizes the **Gymnasium** (formerly OpenAI Gym) API to define RL environments, making it compatible with a wide range of RL algorithms. For agent training and implementation, `flyrl` heavily relies on the **stable-baselines3** library, which provides pre-built RL algorithms and utilities.

## 2. Core Components and Their Roles

### Simulation Environment (`flyrl/environment.py`)

The simulation environment is the central piece that connects the agent with the flight simulation.
*   **JSBSim Wrapping:** The `JsbSimEnv` class acts as a Gymnasium-compliant wrapper around JSBSim. It handles the lifecycle of the simulation, including initialization, stepping through time, and resetting.
*   **State Management:** It manages the simulation state, translating data from JSBSim into observations that the RL agent can understand.
*   **Task and Agent Interaction:** `JsbSimEnv` is instantiated with a specific `Task` that defines the RL problem. It receives actions from the agent, passes them to the task and underlying simulation, and returns the new state, reward, and termination status.
*   **`NoFGJsbSimEnv`:** A specialized version, `NoFGJsbSimEnv`, is provided for scenarios where FlightGear visualization is not required (e.g., during headless training), which can prevent network issues related to repeated socket creation.

### Aircraft (`flyrl/aircraft.py`)

This module defines the aircraft types available for simulation.
*   **Definition:** Aircraft are defined using a `namedtuple` called `Aircraft`, which stores crucial identifiers like the `jsbsim_id` (used by JSBSim to load the model), `flightgear_id` (for FlightGear visualization), a human-readable `name`, and `cruise_speed_kts`.
*   **Usage:** These `Aircraft` objects are used by the simulation environment and tasks to specify which aircraft model to load and to set relevant initial conditions (e.g., initial speed based on cruise speed).
*   **Available Aircraft:** Examples include the `cessna172P`, `f15`, `a320`, and `rascal`.

### Tasks (`flyrl/tasks.py`, `flyrl/basic_tasks.py`)

Tasks define the specific goals and rules for the RL agent within the simulation.
*   **Role:** They encapsulate the logic for state representation, action space definition, initial conditions, episode termination conditions, and reward calculation.
*   **Examples:**
    *   `HeadingControlTask`: Agent learns to maintain steady, level flight at a specific heading.
    *   `TurnHeadingControlTask`: Agent learns to turn from a random initial heading to a random target heading.
    *   (Inferred potential tasks based on file names or general RL applications): `WaypointTask` (navigating a series of waypoints), `DogfightTask` (air combat maneuvers, suggested by environment names like "DogfightRascal").
*   **Task Definition Approaches:**
    *   **`FlightTask` (in `flyrl/tasks.py`):** This provides a more structured way to define tasks. It uses an `Assessor` object, which in turn uses `RewardComponent`s, to calculate rewards. This allows for modular and potentially complex reward shaping.
    *   **`BaseFlightTask` (in `flyrl/basic_tasks.py`):** This offers a simpler approach where reward calculation is done directly within a `calculate_reward` method in the task class itself. It also includes easier integration of an autopilot.

### Rewards (`flyrl/rewards.py`)

This module defines how rewards are structured and calculated, which is crucial for guiding the RL agent's learning process.
*   **Reward Object:** The `Reward` class stores reward elements, separating them into `base_reward_elements` (for task success) and `shaping_reward_elements` (for guiding behavior).
*   **`RewardComponent`s:** Rewards are built from various `RewardComponent` classes. These components calculate a normalized "potential" based on specific aspects of the aircraft's state relative to the task goals (e.g., `AsymptoticErrorComponent` for errors in altitude or heading, `AngularAsymptoticErrorComponent` for heading errors specifically).
*   **Assessors:** `flyrl.assessors.AssessorImpl` (and related classes) aggregate these components to compute the final reward for the agent. This system supports potential-based reward shaping, which can help in learning complex behaviors.

### Simulation Backend (`flyrl/simulation.py`, `flyrl/ap_simulation.py`)

These modules handle the direct interaction with the underlying flight simulation engines.
*   **`Simulation` Class (`flyrl/simulation.py`):**
    *   This class is a Python wrapper around JSBSim's `FGFDMExec`.
    *   It manages loading JSBSim aircraft models, setting initial conditions, running simulation steps, and providing an interface for getting and setting JSBSim properties.
*   **`AP_Simulation` Class (`flyrl/ap_simulation.py`):**
    *   This class provides an interface to ArduPilot (a popular open-source autopilot software suite) using the `dronekit` library.
    *   It allows `flyrl` to interact with ArduPilot SITL (Software-In-The-Loop) or HITL (Hardware-In-The-Loop) simulations.
    *   It maps `flyrl.properties` to DroneKit vehicle attributes and RC channel overrides, enabling state retrieval and control command transmission to an ArduPilot-controlled vehicle.

## 3. Agent Implementation and Training (`flyrl/agents/`)

This part of the framework focuses on the RL agents themselves and how they are trained.
*   **`stable-baselines3` Usage:**
    *   `flyrl` leverages `stable-baselines3` for providing RL algorithms. Scripts show the use or import of PPO (Proximal Policy Optimization), SAC (Soft Actor-Critic), and RecurrentPPO (for tasks requiring memory).
    *   Agents are typically instances of these SB3 models (e.g., `PPO("MlpPolicy", env, ...)`).
*   **`wandb` Integration:**
    *   Weights & Biases (`wandb`) is used extensively for experiment tracking, logging metrics (rewards, episode lengths, gradients, etc.), and saving model artifacts.
    *   The `WandbCallback` from `wandb.integration.sb3` integrates SB3's logging with W&B.
*   **Example Training Scripts:**
    *   `flyrl/agents/train_basic.py` demonstrates a typical training setup:
        1.  Configuration of hyperparameters (policy type, timesteps, environment name, learning rate).
        2.  Initialization of a `wandb` run.
        3.  Creation of a Gymnasium environment (e.g., `DogfightRascal-v0`) using `gym.make()`, often wrapped with `Monitor` and `DummyVecEnv`.
        4.  Instantiation of an SB3 agent (e.g., PPO).
        5.  Training the agent using `model.learn()`, with the `WandbCallback`.
        6.  Saving the trained model.
*   **Pre-trained Model Storage:**
    *   Trained models are saved as `.zip` files (standard for SB3).
    *   The convention observed is `models/<wandb_run_name>/finish.zip` (e.g., `models/deft-elevator-18/finish.zip`), where `<wandb_run_name>` is automatically generated by W&B, ensuring traceability of models to their respective training runs.

## 4. Data and Configuration Files

Various external files are used for data logging, simulation setup, and visualization.
*   **Flight Logs (`.csv` files like `flight_data.csv`, `flight_data2.csv`, `flight_data3.csv`):**
    *   These files store timestamped flight parameters, including Latitude, Longitude, Altitude, Roll, Pitch, Yaw, and sometimes Heading.
    *   They are likely used for post-flight analysis, debugging, generating visualizations, or potentially for data-driven approaches like imitation learning.
    *   Variations in columns (e.g., presence of 'Heading') and altitude scales suggest different logging configurations or flight scenarios.
*   **JSBSim Initial Conditions (`.xml` files like `basic_ic.xml`, `rascal_reset.xml`):**
    *   These XML files are used by JSBSim to define the starting state of a simulation.
    *   `basic_ic.xml`, for example, sets initial altitude (3000 ft), speed (~100 m/s), position (over University of Bath), level attitude, and engines running.
    *   `rascal_reset.xml` (name implies) likely provides specific reset conditions for the Rascal aircraft model.
*   **FlightGear Configuration (`flightgear.xml`):**
    *   This XML file configures JSBSim to output data to FlightGear for 3D visualization.
    *   It specifies UDP as the protocol, `localhost` as the target, and a data rate (e.g., 60 Hz).
    *   A merge conflict was noted in this file regarding the UDP `port` number (5503 vs. 5500), indicating different settings might be active or have been used in different development branches. This port must match FlightGear's listening port.

## 5. Integration with External Tools

`flyrl` integrates several key external tools:
*   **JSBSim:** The core flight dynamics engine. It performs the physics calculations for aircraft movement and state updates based on aerodynamic models and control inputs. `flyrl` wraps and controls JSBSim instances.
*   **FlightGear:** An optional, open-source 3D flight simulator used for high-fidelity visualization. JSBSim sends data to FlightGear via UDP, allowing users to see the aircraft's behavior in a realistic environment. FlightGear's own FDM is bypassed.
*   **ArduPilot:** `flyrl` supports integration with the ArduPilot autopilot software suite via the `APSimulation` class (using `dronekit`) and `APEnv` (custom ArduPilot environment in `ap_environment.py`). This allows for more advanced tasks involving autopilot interaction, SITL, or HITL setups.

## 6. Extensibility

The framework is designed to be extensible in several ways:
*   **Defining New Flight Tasks:**
    *   Users can create new task classes by subclassing `flyrl.tasks.FlightTask` (for structured, assessor-based rewards) or `flyrl.basic_tasks.BaseFlightTask` (for simpler, direct reward calculation).
    *   This involves defining state/action spaces, initial conditions, termination logic, reward functions, and any custom properties.
    *   New tasks must be registered with `gymnasium.register()` to be accessible via `gym.make()`.
*   **Adding New Aircraft Models:**
    *   Requires creating JSBSim XML definition files for the new aircraft (aerodynamics, systems, etc.).
    *   An `Aircraft` object must be added to `flyrl/aircraft.py`, specifying the `jsbsim_id` and other relevant details.
*   **Implementing New RL Agents or Training Routines:**
    *   **Using different `stable-baselines3` algorithms:** Easily done by modifying training scripts to import and use a different SB3 agent.
    *   **Custom SB3 policy networks:** Users can define custom network architectures and pass them to SB3 agents.
    *   **Entirely new RL algorithms:** Requires more significant effort, implementing the algorithm and a custom training loop that interacts with the `flyrl` Gymnasium environments.
    *   **Training routine modifications:** Can involve hyperparameter tuning, custom SB3 callbacks, environment wrappers, or curriculum learning strategies.

## 7. Potential Use Cases

`flyrl` is well-suited for a variety of applications in aerospace and AI research:
*   **Research in Autonomous Flight:** Developing and evaluating RL agents for complex flight control tasks (e.g., maneuvering, navigation, aerobatics, formation flying).
*   **Pilot Training Aids:** Creating intelligent systems that can demonstrate maneuvers or act as adaptive opponents/instructors.
*   **Benchmarking RL Algorithms:** Providing a standardized set of challenging flight environments for comparing the performance of different RL algorithms.
*   **UAS/Drone Autonomy:** Developing autonomous control systems for unmanned aerial systems.
*   **Fault-Tolerant Control Systems:** Training agents that can adapt to simulated aircraft system failures.

## 8. Project Structure Overview

The project is organized into several key directories:
*   **`flyrl/`:** The main source code directory for the framework.
    *   `flyrl/agents/`: Contains scripts for training and testing RL agents.
    *   `flyrl/tests/`: Contains unit and integration tests for the framework components.
    *   Other Python files in `flyrl/` define core components like environments, tasks, aircraft, simulation wrappers, rewards, properties, and visualizers.
    *   XML files (e.g., `basic_ic.xml`, `flightgear.xml`) for JSBSim and FlightGear configuration are also located here.
*   **`models/`:** Stores pre-trained agent models, typically organized by W&B run names (e.g., `models/deft-elevator-18/finish.zip`).
*   **Root Directory:** Contains project-level files like `README.md`, `setup.py`, license information, and potentially some data logs or configuration files that might be moved or are for general testing.
```
