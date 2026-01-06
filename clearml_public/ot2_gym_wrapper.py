import gymnasium as gym
import numpy as np
import json
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
from sim_class import Simulation


class OT2GymEnv(gym.Env):
    """
    Gymnasium environment for Opentrons OT-2 robot control using the Simulation class.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 500,
        action_scale: float = 0.1,
        success_threshold: float = 0.001,
        num_agents: int = 1,
        working_envelope_path: str = "working_envelope.json",
        require_drop: bool = True,
        # Reward function parameters
        distance_weight: float = 10.0,
        progress_weight: float = 50.0,
        success_bonus: float = 100.0,
        time_penalty: float = 0.01,
        workspace_penalty: float = 10.0,
        drop_bonus_perfect: float = 50.0,
        drop_bonus_good: float = 20.0,
        drop_penalty_poor: float = 10.0,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.success_threshold = success_threshold
        self.num_agents = num_agents
        self.require_drop = require_drop

        # Reward function parameters
        self.distance_weight = distance_weight
        self.progress_weight = progress_weight
        self.success_bonus = success_bonus
        self.time_penalty = time_penalty
        self.workspace_penalty = workspace_penalty
        self.drop_bonus_perfect = drop_bonus_perfect
        self.drop_bonus_good = drop_bonus_good
        self.drop_penalty_poor = drop_penalty_poor

        # working envelope
        self.workspace_bounds = self._load_working_envelope(working_envelope_path)

        # simulation instance
        render = (render_mode == "human")
        rgb_array = (render_mode == "rgb_array")
        self.sim = Simulation(
            num_agents=num_agents,
            render=render,
            rgb_array=rgb_array
        )

        # action space: [vx, vy, vz, drop]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # observation space: [current_tip_pos(3), target_pos(3), velocity(3), has_dropped(1)]
        obs_dim = 3 + 3 + 3 + 1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # episode tracking
        self.current_step = 0
        self.target_position = None
        self.previous_distance = None
        self.initial_num_droplets = 0

        # initial robot and specimen IDs
        self.robot_id = self.sim.robotIds[0]
        self.specimen_id = self.sim.specimenIds[0]

    def _load_working_envelope(self, json_path: str) -> Dict[str, Tuple[float, float]]:
        try:
            with open(json_path, 'r') as f:
                corners = json.load(f)
        except FileNotFoundError:
            return {
                'x': (-0.1873, 0.2533),
                'y': (-0.1708, 0.2199),
                'z': (0.1693, 0.2898)
            }

        # extract corner positions
        positions = np.array([corners[str(i)] for i in range(1, 9)])

        # min and max
        bounds = {
            'x': (positions[:, 0].min(), positions[:, 0].max()),
            'y': (positions[:, 1].min(), positions[:, 1].max()),
            'z': (positions[:, 2].min(), positions[:, 2].max())
        }

        return bounds

    def _get_obs(self) -> np.ndarray:
        # get states
        states = self.sim.get_states()
        robot_key = f'robotId_{self.robot_id}'

        # extract pipette position
        tip_position = np.array(states[robot_key]['pipette_position'])

        # extract joint velocities
        velocities = np.array([
            states[robot_key]['joint_states'][f'joint_{i}']['velocity']
            for i in range(3)
        ])

        # if droplet was dropped
        has_dropped = self._check_has_dropped()

        # construct observation
        obs = np.concatenate([
            tip_position,
            self.target_position,
            velocities,
            np.array([1.0 if has_dropped else 0.0])
        ]).astype(np.float32)

        return obs

    def _check_has_dropped(self) -> bool:
        current_num_droplets = len(self.sim.sphereIds)
        return current_num_droplets > self.initial_num_droplets

    def _get_drop_position(self) -> Optional[np.ndarray]:
        specimen_key = f'specimenId_{self.specimen_id}'
        if specimen_key in self.sim.droplet_positions:
            positions = self.sim.droplet_positions[specimen_key]
            if len(positions) > 0:
                return np.array(positions[-1])
        return None

    def _get_info(self) -> Dict[str, Any]:
        tip_position = self._get_tip_position()
        distance = np.linalg.norm(tip_position - self.target_position)
        has_dropped = self._check_has_dropped()

        info = {
            "distance_to_target": distance,
            "previous_distance": self.previous_distance,
            "current_step": self.current_step,
            "tip_position": tip_position.tolist(),
            "target_position": self.target_position.tolist(),
            "has_dropped": has_dropped,
            "num_droplets": len(self.sim.sphereIds),
            "in_workspace": self._in_workspace(tip_position)
        }

        # get actual drop position
        drop_position = self._get_drop_position()
        if drop_position is not None:
            drop_error = np.linalg.norm(drop_position - self.target_position)
            info["drop_error"] = drop_error
            info["drop_position"] = drop_position.tolist()

            # success criteria
            if self.require_drop:
                info["is_success"] = drop_error < self.success_threshold
            else:
                info["is_success"] = distance < self.success_threshold
        else:
            # no drop yet
            if self.require_drop:
                info["is_success"] = False
            else:
                info["is_success"] = distance < self.success_threshold

        return info

    def _compute_reward(self, info: Dict[str, Any], action: np.ndarray) -> float:
        distance = info["distance_to_target"]
        previous_distance = info.get("previous_distance", distance)

        reward = 0.0

        # 1. Distance-based reward
        reward -= distance * self.distance_weight

        # 2. Progress reward
        if previous_distance is not None:
            progress = previous_distance - distance
            reward += progress * self.progress_weight

        # 3. Success bonus
        if info.get("is_success", False):
            reward += self.success_bonus

        # 4. Drop accuracy rewards
        if "drop_error" in info:
            drop_error = info["drop_error"]
            if drop_error < 0.001:
                reward += self.drop_bonus_perfect
            elif drop_error < 0.003:
                reward += self.drop_bonus_good
            else:
                reward -= self.drop_penalty_poor

        # 5. Workspace boundary penalty
        if not info.get("in_workspace", True):
            reward -= self.workspace_penalty

        # 6. Time penalty
        reward -= self.time_penalty

        return reward

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # reset simulation
        self.sim.reset(num_agents=self.num_agents)

        # update robot and specimen IDs after reset
        self.robot_id = self.sim.robotIds[0]
        self.specimen_id = self.sim.specimenIds[0]

        # track initial droplet count
        self.initial_num_droplets = len(self.sim.sphereIds)

        # set to home position
        home_x = (self.workspace_bounds['x'][0] + self.workspace_bounds['x'][1]) / 2
        home_y = (self.workspace_bounds['y'][0] + self.workspace_bounds['y'][1]) / 2
        home_z = (self.workspace_bounds['z'][0] + self.workspace_bounds['z'][1]) / 2

        self.sim.set_start_position(home_x, home_y, home_z)

        # sample random target position within workspace
        if options and 'target_position' in options:
            self.target_position = np.array(options['target_position'])
        else:
            self.target_position = np.array([
                self.np_random.uniform(*self.workspace_bounds['x']),
                self.np_random.uniform(*self.workspace_bounds['y']),
                self.np_random.uniform(*self.workspace_bounds['z'])
            ])

        # reset episode tracking
        self.current_step = 0
        self.previous_distance = np.linalg.norm(
            self._get_tip_position() - self.target_position
        )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # scale velocity actions
        scaled_velocity = action[:3] * self.action_scale

        # convert drop action: trigger if > 0.0
        drop_action = 1 if action[3] > 0.0 else 0

        # format action for simulation: [[vx, vy, vz, drop]]
        sim_action = [[scaled_velocity[0], scaled_velocity[1], scaled_velocity[2], drop_action]]

        # run simulation for one step
        self.sim.run(sim_action, num_steps=1)

        # get new observation and info
        observation = self._get_obs()
        info = self._get_info()

        # compute reward using current and previous state
        reward = self._compute_reward(info, action)

        # update previous_distance for next step
        self.previous_distance = info["distance_to_target"]

        # check termination conditions
        terminated = info["is_success"]
        truncated = self.current_step >= self.max_steps

        self.current_step += 1

        return observation, reward, terminated, truncated, info

    def _get_tip_position(self) -> np.ndarray:
        states = self.sim.get_states()
        robot_key = f'robotId_{self.robot_id}'
        return np.array(states[robot_key]['pipette_position'])

    def _in_workspace(self, position: np.ndarray) -> bool:
        return (
            self.workspace_bounds['x'][0] <= position[0] <= self.workspace_bounds['x'][1] and
            self.workspace_bounds['y'][0] <= position[1] <= self.workspace_bounds['y'][1] and
            self.workspace_bounds['z'][0] <= position[2] <= self.workspace_bounds['z'][1]
        )

    def render(self):
        if self.render_mode == "rgb_array":
            if hasattr(self.sim, 'current_frame'):
                return self.sim.current_frame
            return None
        return None

    def get_plate_image_path(self) -> str:
        return self.sim.get_plate_image()

    def close(self):
        self.sim.close()
