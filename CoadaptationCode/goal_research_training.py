import argparse
import csv
import json
import math
import os
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.policies import TanhGaussianPolicy


MOTOR_COUNT = 7
SEGMENT_COUNT = 8
OBSERVATION_DIM = 13
ACTION_DIM = MOTOR_COUNT
HIDDEN_SIZES = [256, 256, 256]
DEFAULT_DMAX_CM = 40.0
DEFAULT_REPLAY_SIZE = 100_000
SCALE_WIDTH_BOUNDS = (0.45, 0.90)
SCALE_ANGLE_BOUNDS = (0.0, 30.0)
DESIGN_CONDITION_LABELS = [
    "A_Width_Norm",
    "A_Angle_Norm",
    "B_Width_Norm",
    "B_Angle_Norm",
    "Delta_Width_Norm",
    "Delta_Angle_Norm",
]
TERRAIN_LABELS = ["carpet", "cardboard"]
TERRAIN_CONDITION_LABELS = ["Terrain_Carpet", "Terrain_Cardboard"]
CONDITION_LABELS = DESIGN_CONDITION_LABELS + TERRAIN_CONDITION_LABELS
DESIGN_CONDITION_DIM = len(DESIGN_CONDITION_LABELS)
TERRAIN_CONDITION_DIM = len(TERRAIN_CONDITION_LABELS)
CONDITION_DIM = len(CONDITION_LABELS)
CONDITIONED_OBSERVATION_DIM = OBSERVATION_DIM + CONDITION_DIM
OBSERVATION_LABELS = [
    "Target_Delta_X_Norm",
    "Target_Delta_Y_Norm",
    "Target_Forward_Remaining_Norm",
    "Theta_X_Norm",
    "Theta_Y_Norm",
    "Theta_Z_Norm",
    "Motor1_Norm",
    "Motor2_Norm",
    "Motor3_Norm",
    "Motor4_Norm",
    "Motor5_Norm",
    "Motor6_Norm",
    "Motor7_Norm",
]
MORPHOLOGY_MODULE_PATTERN = ["A", "B", "A", "B", "A", "B", "A", "B"]


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def wrap_to_pi(angle_rad: np.ndarray) -> np.ndarray:
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def _normalize_to_unit(value: float, bounds: Tuple[float, float]) -> float:
    low, high = float(bounds[0]), float(bounds[1])
    if high <= low:
        raise ValueError(f"Invalid normalization bounds: {bounds}.")
    return float(np.clip(2.0 * (float(value) - low) / (high - low) - 1.0, -1.0, 1.0))


def optitrack_xzy_degrees_to_physical_xyz_radians(orientation_deg: Sequence[float]) -> np.ndarray:
    """Map scipy's as_euler('xzy') output to physical [x, y, z] radians."""
    euler_xzy = np.asarray(orientation_deg, dtype=np.float32).reshape(3)
    physical_xyz_deg = np.asarray([euler_xzy[0], euler_xzy[2], euler_xzy[1]], dtype=np.float32)
    return np.deg2rad(physical_xyz_deg).astype(np.float32)


def action_to_motor_counts(
    action: Sequence[float],
    center_count: int = 2048,
    max_degrees: float = 55.0,
    counts_per_degree: float = 4096.0 / 360.0,
) -> np.ndarray:
    action_array = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    if action_array.shape != (MOTOR_COUNT,):
        raise ValueError(f"Expected {MOTOR_COUNT} actions, got shape {action_array.shape}.")
    motor_degrees = action_array * float(max_degrees)
    counts = np.rint(float(center_count) + motor_degrees * float(counts_per_degree)).astype(np.int32)
    min_count = int(round(float(center_count) - float(max_degrees) * float(counts_per_degree)))
    max_count = int(round(float(center_count) + float(max_degrees) * float(counts_per_degree)))
    return np.clip(counts, min_count, max_count)


def motor_counts_to_degrees(
    counts: Sequence[float],
    center_count: int = 2048,
    counts_per_degree: float = 4096.0 / 360.0,
    max_degrees: float = 55.0,
) -> np.ndarray:
    degrees = (np.asarray(counts, dtype=np.float32) - float(center_count)) / float(counts_per_degree)
    return np.clip(degrees, -float(max_degrees), float(max_degrees))


def build_goal_observation(
    start_position_cm: Sequence[float],
    current_position_cm: Sequence[float],
    start_orientation_rad: Sequence[float],
    current_orientation_rad: Sequence[float],
    motor_angles_deg: Sequence[float],
    target_position_cm: Optional[Sequence[float]] = None,
    use_target_relative_position: bool = False,
    dmax: float = DEFAULT_DMAX_CM,
    motor_limit_deg: float = 55.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    start_pos = np.asarray(start_position_cm, dtype=np.float32).reshape(3)
    curr_pos = np.asarray(current_position_cm, dtype=np.float32).reshape(3)
    start_ori = np.asarray(start_orientation_rad, dtype=np.float32).reshape(3)
    curr_ori = np.asarray(current_orientation_rad, dtype=np.float32).reshape(3)
    motor_deg = np.asarray(motor_angles_deg, dtype=np.float32).reshape(MOTOR_COUNT)

    if use_target_relative_position:
        if target_position_cm is None:
            raise ValueError("target_position_cm is required for target-relative observations.")
        target_pos = np.asarray(target_position_cm, dtype=np.float32).reshape(3)
        delta_x = float(target_pos[0] - curr_pos[0])
        delta_y = float(target_pos[1] - curr_pos[1])
        # Forward is negative z, so this is the remaining distance to the target.
        delta_forward = float(curr_pos[2] - target_pos[2])
    else:
        target_pos = None
        delta_x = float(curr_pos[0] - start_pos[0])
        delta_y = float(curr_pos[1] - start_pos[1])
        delta_forward = float(start_pos[2] - curr_pos[2])
    theta = wrap_to_pi(curr_ori - start_ori).astype(np.float32)

    dmax = max(float(dmax), 1e-6)
    forward_scale = dmax
    motor_limit_deg = max(float(motor_limit_deg), 1e-6)
    obs = np.concatenate(
        [
            np.asarray([delta_x / dmax, delta_y / dmax, delta_forward / forward_scale], dtype=np.float32),
            theta / np.pi,
            motor_deg / motor_limit_deg,
        ]
    )
    obs = np.clip(obs, -1.0, 1.0).astype(np.float32)
    raw = {
        "delta_x_cm": delta_x,
        "delta_y_cm": delta_y,
        "delta_forward_cm": delta_forward,
        "forward_scale_cm": float(forward_scale),
        "position_observation_mode": "target_relative" if use_target_relative_position else "start_relative",
        "theta_x_rad": float(theta[0]),
        "theta_y_rad": float(theta[1]),
        "theta_z_rad": float(theta[2]),
    }
    if target_pos is not None:
        raw.update(
            {
                "target_x_cm": float(target_pos[0]),
                "target_y_cm": float(target_pos[1]),
                "target_z_cm": float(target_pos[2]),
            }
        )
    for idx, value in enumerate(motor_deg, start=1):
        raw[f"motor{idx}_deg"] = float(value)
    return obs, raw


def compute_goal_reward(
    start_z_cm: float,
    current_z_cm: float,
    target_z_cm: float,
    yaw_rad: Optional[float] = None,
    theta_y_rad: Optional[float] = None,
    dmax: float = DEFAULT_DMAX_CM,
    success_bonus: float = 0.0,
    time_penalty: float = 0.0,
    terminated_by_target: bool = False,
) -> Tuple[float, Dict[str, float]]:
    if yaw_rad is None:
        if theta_y_rad is None:
            raise ValueError("compute_goal_reward requires yaw_rad.")
        yaw_rad = theta_y_rad
    forward_curr = float(start_z_cm) - float(current_z_cm)
    forward_target = float(start_z_cm) - float(target_z_cm)
    distance_error = abs(forward_target - forward_curr)
    forward_term = math.exp(1.0 - distance_error / max(float(dmax), 1e-6))
    yaw_term = 0.3 - 0.3 * abs(float(yaw_rad))
    success_term = float(success_bonus) if bool(terminated_by_target) else 0.0
    time_term = -abs(float(time_penalty))
    reward = float(forward_term + yaw_term + success_term + time_term)
    yaw_penalty_rad = abs(float(yaw_rad))
    components = {
        "forward_curr_cm": forward_curr,
        "forward_target_cm": forward_target,
        "distance_error_cm": distance_error,
        "forward_reward": float(forward_term),
        "yaw_reward": float(yaw_term),
        "yaw_penalty_rad": yaw_penalty_rad,
        "yaw_rad": float(yaw_rad),
        "success_bonus": success_term,
        "time_penalty": time_term,
        "reward": reward,
    }
    return reward, components


@dataclass(frozen=True)
class MorphologyConfig:
    layout: str
    a_width: float
    a_angle: float
    b_width: float
    b_angle: float

    def as_design_vector(self) -> List[float]:
        return [self.a_width, self.a_angle, self.b_width, self.b_angle]

    def expanded_modules(self) -> List[Dict[str, float]]:
        modules = []
        for module_idx, group in enumerate(MORPHOLOGY_MODULE_PATTERN, start=1):
            if group == "A":
                width = self.a_width
                angle = self.a_angle
            else:
                width = self.b_width
                angle = self.b_angle
            modules.append(
                {
                    "module": module_idx,
                    "group": group,
                    "width_ratio": float(width),
                    "attack_angle_deg": float(angle),
                }
            )
        return modules

    def metadata(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["design_vector"] = self.as_design_vector()
        payload["module_pattern"] = MORPHOLOGY_MODULE_PATTERN
        payload["expanded_modules"] = self.expanded_modules()
        return payload


def build_morphology_config(
    layout: str,
    a_width: float,
    a_angle: float,
    b_width: Optional[float] = None,
    b_angle: Optional[float] = None,
) -> MorphologyConfig:
    layout = str(layout).strip().lower()
    if layout not in {"homogeneous", "heterogeneous_ab"}:
        raise ValueError("layout must be 'homogeneous' or 'heterogeneous_ab'.")
    if layout == "homogeneous":
        b_width = a_width
        b_angle = a_angle
    if b_width is None or b_angle is None:
        raise ValueError("heterogeneous_ab layout requires --b-width and --b-angle.")
    return MorphologyConfig(
        layout=layout,
        a_width=float(a_width),
        a_angle=float(a_angle),
        b_width=float(b_width),
        b_angle=float(b_angle),
    )


def design_condition_from_morphology(morphology: MorphologyConfig) -> np.ndarray:
    width_span = SCALE_WIDTH_BOUNDS[1] - SCALE_WIDTH_BOUNDS[0]
    angle_span = SCALE_ANGLE_BOUNDS[1] - SCALE_ANGLE_BOUNDS[0]
    condition = np.asarray(
        [
            _normalize_to_unit(morphology.a_width, SCALE_WIDTH_BOUNDS),
            _normalize_to_unit(morphology.a_angle, SCALE_ANGLE_BOUNDS),
            _normalize_to_unit(morphology.b_width, SCALE_WIDTH_BOUNDS),
            _normalize_to_unit(morphology.b_angle, SCALE_ANGLE_BOUNDS),
            np.clip((float(morphology.b_width) - float(morphology.a_width)) / max(width_span, 1e-6), -1.0, 1.0),
            np.clip((float(morphology.b_angle) - float(morphology.a_angle)) / max(angle_span, 1e-6), -1.0, 1.0),
        ],
        dtype=np.float32,
    )
    return condition


def terrain_condition(terrain: str) -> np.ndarray:
    normalized = str(terrain).strip().lower()
    if normalized not in TERRAIN_LABELS:
        raise ValueError(f"Unknown terrain '{terrain}'. Expected one of {TERRAIN_LABELS}.")
    condition = np.zeros(TERRAIN_CONDITION_DIM, dtype=np.float32)
    condition[TERRAIN_LABELS.index(normalized)] = 1.0
    return condition


def build_condition_vector(morphology: MorphologyConfig, terrain: str) -> np.ndarray:
    return np.concatenate([design_condition_from_morphology(morphology), terrain_condition(terrain)]).astype(np.float32)


def build_conditioned_observation(robot_observation: Sequence[float], condition: Sequence[float]) -> np.ndarray:
    robot_obs = np.asarray(robot_observation, dtype=np.float32).reshape(OBSERVATION_DIM)
    condition_array = np.asarray(condition, dtype=np.float32).reshape(CONDITION_DIM)
    return np.concatenate([robot_obs, condition_array]).astype(np.float32)


class GoalReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_dim: int = OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        condition_dim: int = CONDITION_DIM,
    ):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.condition_dim = int(condition_dim)
        self.conditioned_obs_dim = self.obs_dim + self.condition_dim
        self.robot_observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.robot_next_observations = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.conditions = np.zeros((self.capacity, self.condition_dim), dtype=np.float32)
        self.next_conditions = np.zeros((self.capacity, self.condition_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.terminals = np.zeros((self.capacity, 1), dtype=np.float32)
        self.terrain_ids = np.full((self.capacity, 1), -1, dtype=np.int32)
        self.design_ids = np.full((self.capacity, 1), -1, dtype=np.int32)
        self.top = 0
        self.size = 0

    def add_sample(
        self,
        observation: Sequence[float],
        action: Sequence[float],
        reward: float,
        next_observation: Sequence[float],
        terminal: bool,
        condition: Optional[Sequence[float]] = None,
        next_condition: Optional[Sequence[float]] = None,
        terrain_id: int = -1,
        design_id: int = -1,
    ) -> None:
        robot_observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        robot_next_observation = np.asarray(next_observation, dtype=np.float32).reshape(-1)
        if condition is None and robot_observation.size == self.conditioned_obs_dim:
            conditioned_observation = robot_observation.astype(np.float32)
            robot_observation = conditioned_observation[: self.obs_dim]
            condition_array = conditioned_observation[self.obs_dim :]
        else:
            robot_observation = robot_observation.reshape(self.obs_dim).astype(np.float32)
            condition_array = np.zeros(self.condition_dim, dtype=np.float32)
            if condition is not None:
                condition_array = np.asarray(condition, dtype=np.float32).reshape(self.condition_dim)

        if next_condition is None and robot_next_observation.size == self.conditioned_obs_dim:
            conditioned_next_observation = robot_next_observation.astype(np.float32)
            robot_next_observation = conditioned_next_observation[: self.obs_dim]
            next_condition_array = conditioned_next_observation[self.obs_dim :]
        else:
            robot_next_observation = robot_next_observation.reshape(self.obs_dim).astype(np.float32)
            next_condition_array = condition_array.copy()
            if next_condition is not None:
                next_condition_array = np.asarray(next_condition, dtype=np.float32).reshape(self.condition_dim)

        self.robot_observations[self.top] = robot_observation
        self.robot_next_observations[self.top] = robot_next_observation
        self.conditions[self.top] = condition_array
        self.next_conditions[self.top] = next_condition_array
        self.actions[self.top] = np.asarray(action, dtype=np.float32)
        self.rewards[self.top, 0] = float(reward)
        self.terminals[self.top, 0] = float(bool(terminal))
        self.terrain_ids[self.top, 0] = int(terrain_id)
        self.design_ids[self.top, 0] = int(design_id)
        self.top = (self.top + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def random_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        indices = np.random.choice(self.size, size=int(batch_size), replace=True)
        observations = np.concatenate([self.robot_observations[indices], self.conditions[indices]], axis=1)
        next_observations = np.concatenate(
            [self.robot_next_observations[indices], self.next_conditions[indices]],
            axis=1,
        )
        return {
            "observations": observations,
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "terminals": self.terminals[indices],
            "next_observations": next_observations,
            "robot_observations": self.robot_observations[indices],
            "robot_next_observations": self.robot_next_observations[indices],
            "conditions": self.conditions[indices],
            "next_conditions": self.next_conditions[indices],
            "terrain_ids": self.terrain_ids[indices],
            "design_ids": self.design_ids[indices],
        }

    def sample_robot_observations(self, batch_size: int) -> np.ndarray:
        if self.size <= 0:
            raise RuntimeError("Cannot sample from an empty replay buffer.")
        indices = np.random.choice(self.size, size=int(batch_size), replace=True)
        return self.robot_observations[indices]

    def clear(self) -> None:
        self.top = 0
        self.size = 0

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                capacity=np.asarray([self.capacity], dtype=np.int64),
                obs_dim=np.asarray([self.obs_dim], dtype=np.int64),
                action_dim=np.asarray([self.action_dim], dtype=np.int64),
                condition_dim=np.asarray([self.condition_dim], dtype=np.int64),
                top=np.asarray([self.top], dtype=np.int64),
                size=np.asarray([self.size], dtype=np.int64),
                robot_observations=self.robot_observations,
                robot_next_observations=self.robot_next_observations,
                conditions=self.conditions,
                next_conditions=self.next_conditions,
                actions=self.actions,
                rewards=self.rewards,
                terminals=self.terminals,
                terrain_ids=self.terrain_ids,
                design_ids=self.design_ids,
            )
        tmp_path.replace(path)

    def load_npz(self, path: Path) -> None:
        data = np.load(path)
        saved_capacity = int(data["capacity"][0])
        if saved_capacity != self.capacity:
            self.capacity = saved_capacity
        self.obs_dim = int(data["obs_dim"][0])
        self.action_dim = int(data["action_dim"][0])
        self.condition_dim = int(data["condition_dim"][0])
        self.conditioned_obs_dim = self.obs_dim + self.condition_dim
        self.top = int(data["top"][0])
        self.size = int(data["size"][0])
        self.robot_observations = data["robot_observations"].astype(np.float32)
        self.robot_next_observations = data["robot_next_observations"].astype(np.float32)
        self.conditions = data["conditions"].astype(np.float32)
        self.next_conditions = data["next_conditions"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)
        self.rewards = data["rewards"].astype(np.float32)
        self.terminals = data["terminals"].astype(np.float32)
        self.terrain_ids = data["terrain_ids"].astype(np.int32)
        self.design_ids = data["design_ids"].astype(np.int32)

    def __len__(self) -> int:
        return self.size


class GoalSACAgent:
    def __init__(
        self,
        obs_dim: int = CONDITIONED_OBSERVATION_DIM,
        action_dim: int = ACTION_DIM,
        hidden_sizes: Sequence[int] = HIDDEN_SIZES,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        alpha_init: float = 0.01,
        target_entropy: float = -float(ACTION_DIM),
        grad_clip_value: float = 1.0,
        device: Optional[str] = None,
    ):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_sizes = list(hidden_sizes)
        self.lr = float(lr)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.alpha_init = float(alpha_init)
        self.target_entropy = float(target_entropy)
        self.grad_clip_value = float(grad_clip_value)
        self.device = torch.device(device or "cpu")
        self.total_updates = 0

        self.policy = TanhGaussianPolicy(
            hidden_sizes=self.hidden_sizes,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
        ).to(self.device)
        self.qf1 = ConcatMlp(
            hidden_sizes=self.hidden_sizes,
            input_size=self.obs_dim + self.action_dim,
            output_size=1,
        ).to(self.device)
        self.qf2 = ConcatMlp(
            hidden_sizes=self.hidden_sizes,
            input_size=self.obs_dim + self.action_dim,
            output_size=1,
        ).to(self.device)
        self.target_qf1 = ConcatMlp(
            hidden_sizes=self.hidden_sizes,
            input_size=self.obs_dim + self.action_dim,
            output_size=1,
        ).to(self.device)
        self.target_qf2 = ConcatMlp(
            hidden_sizes=self.hidden_sizes,
            input_size=self.obs_dim + self.action_dim,
            output_size=1,
        ).to(self.device)
        self.target_qf1.load_state_dict(self.qf1.state_dict())
        self.target_qf2.load_state_dict(self.qf2.state_dict())

        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.qf1_optimizer = torch.optim.Adam(self.qf1.parameters(), lr=self.lr)
        self.qf2_optimizer = torch.optim.Adam(self.qf2.parameters(), lr=self.lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.log_alpha.data.fill_(math.log(max(self.alpha_init, 1e-8)))
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self.last_diagnostics: Dict[str, float] = {}

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().detach().cpu().item())

    def _tensor_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in batch.items()
        }

    def _policy_outputs_tuple(
        self,
        obs: torch.Tensor,
        deterministic: bool,
        return_log_prob: bool,
        reparameterize: bool,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        try:
            outputs = self.policy(
                obs,
                reparameterize=reparameterize,
                deterministic=deterministic,
                return_log_prob=return_log_prob,
            )
        except TypeError:
            return None
        if not isinstance(outputs, (tuple, list)):
            return None
        action = outputs[0]
        mean = outputs[1] if len(outputs) > 1 else action
        log_std = outputs[2] if len(outputs) > 2 else torch.zeros_like(action)
        log_pi = outputs[3] if return_log_prob and len(outputs) > 3 else None
        return action, mean, log_std, log_pi

    def _policy_sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tuple_outputs = self._policy_outputs_tuple(
            obs,
            deterministic=False,
            return_log_prob=True,
            reparameterize=True,
        )
        if tuple_outputs is not None:
            actions, mean, log_std, log_pi = tuple_outputs
            if log_pi is None:
                raise RuntimeError("Tuple-style TanhGaussianPolicy did not return log probability.")
            if log_pi.dim() == 1:
                log_pi = log_pi.unsqueeze(-1)
            elif log_pi.dim() > 1 and log_pi.shape[-1] != 1:
                log_pi = log_pi.sum(dim=-1, keepdim=True)
            return actions, log_pi, mean, log_std

        dist = self.policy(obs)
        if hasattr(dist, "rsample_and_logprob"):
            actions, log_pi = dist.rsample_and_logprob()
        else:
            actions = dist.rsample()
            log_pi = dist.log_prob(actions)
        if log_pi.dim() == 1:
            log_pi = log_pi.unsqueeze(-1)
        elif log_pi.dim() > 1 and log_pi.shape[-1] != 1:
            log_pi = log_pi.sum(dim=-1, keepdim=True)
        mean = getattr(dist, "normal_mean", getattr(dist, "mean", actions))
        std = getattr(dist, "normal_std", None)
        if std is None:
            std = torch.ones_like(actions)
        log_std = torch.log(torch.clamp(std, min=1e-6))
        return actions, log_pi, mean, log_std

    def act(self, observation: Sequence[float], deterministic: bool = False) -> np.ndarray:
        self.policy.eval()
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            tuple_outputs = self._policy_outputs_tuple(
                obs,
                deterministic=deterministic,
                return_log_prob=False,
                reparameterize=False,
            )
            if tuple_outputs is not None:
                action = tuple_outputs[0]
            else:
                dist = self.policy(obs)
                if deterministic:
                    if hasattr(dist, "mle_estimate"):
                        action = dist.mle_estimate()
                    else:
                        action = getattr(dist, "mean", dist.sample())
                else:
                    action = dist.rsample() if hasattr(dist, "rsample") else dist.sample()
        return np.clip(action.squeeze(0).detach().cpu().numpy().astype(np.float32), -1.0, 1.0)

    def act_batch(self, observations: np.ndarray, deterministic: bool = False) -> np.ndarray:
        self.policy.eval()
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            tuple_outputs = self._policy_outputs_tuple(
                obs,
                deterministic=deterministic,
                return_log_prob=False,
                reparameterize=False,
            )
            if tuple_outputs is not None:
                action = tuple_outputs[0]
            else:
                dist = self.policy(obs)
                if deterministic:
                    if hasattr(dist, "mle_estimate"):
                        action = dist.mle_estimate()
                    else:
                        action = getattr(dist, "mean", dist.sample())
                else:
                    action = dist.rsample() if hasattr(dist, "rsample") else dist.sample()
        return np.clip(action.detach().cpu().numpy().astype(np.float32), -1.0, 1.0)

    def q_values(self, observations: np.ndarray, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.qf1.eval()
        self.qf2.eval()
        obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        act = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q1 = self.qf1(obs, act).detach().cpu().numpy()
            q2 = self.qf2(obs, act).detach().cpu().numpy()
        return q1, q2

    def train_step(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        tensors = self._tensor_batch(batch)
        obs = tensors["observations"]
        actions = tensors["actions"]
        rewards = tensors["rewards"]
        terminals = tensors["terminals"]
        next_obs = tensors["next_observations"]

        new_actions, log_pi, policy_mean, policy_log_std = self._policy_sample(obs)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        torch.nn.utils.clip_grad_value_([self.log_alpha], self.grad_clip_value)
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp().detach()

        q_new_actions = torch.min(self.qf1(obs, new_actions), self.qf2(obs, new_actions))
        policy_loss = (alpha * log_pi - q_new_actions).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), self.grad_clip_value)
        self.policy_optimizer.step()

        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        with torch.no_grad():
            next_actions, next_log_pi, _, _ = self._policy_sample(next_obs)
            target_q = torch.min(
                self.target_qf1(next_obs, next_actions),
                self.target_qf2(next_obs, next_actions),
            ) - alpha * next_log_pi
            q_target = rewards + (1.0 - terminals) * self.gamma * target_q

        qf1_loss = F.mse_loss(q1_pred, q_target)
        qf2_loss = F.mse_loss(q2_pred, q_target)

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        torch.nn.utils.clip_grad_value_(self.qf1.parameters(), self.grad_clip_value)
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        torch.nn.utils.clip_grad_value_(self.qf2.parameters(), self.grad_clip_value)
        self.qf2_optimizer.step()

        self._soft_update(self.qf1, self.target_qf1)
        self._soft_update(self.qf2, self.target_qf2)
        self.total_updates += 1

        diagnostics = {
            "qf1_loss": float(qf1_loss.detach().cpu().item()),
            "qf2_loss": float(qf2_loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "alpha": self.alpha,
            "alpha_loss": float(alpha_loss.detach().cpu().item()),
            "target_entropy_internal": float(self.target_entropy),
            "target_entropy_paper": float(abs(self.target_entropy)),
            "q_target_mean": float(q_target.detach().mean().cpu().item()),
            "q_target_std": float(q_target.detach().std(unbiased=False).cpu().item()),
            "policy_mean_abs": float(policy_mean.detach().abs().mean().cpu().item()),
            "policy_log_std_mean": float(policy_log_std.detach().mean().cpu().item()),
            "train_update": int(self.total_updates),
        }
        self.last_diagnostics = diagnostics
        return diagnostics

    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        with torch.no_grad():
            for source_param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.mul_(1.0 - self.tau)
                target_param.data.add_(self.tau * source_param.data)

    def state_dict(self) -> Dict[str, object]:
        return {
            "policy": self.policy.state_dict(),
            "qf1": self.qf1.state_dict(),
            "qf2": self.qf2.state_dict(),
            "target_qf1": self.target_qf1.state_dict(),
            "target_qf2": self.target_qf2.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "qf1_optimizer": self.qf1_optimizer.state_dict(),
            "qf2_optimizer": self.qf2_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "total_updates": self.total_updates,
            "hyperparameters": self.hyperparameters(),
        }

    def _reset_optimizers(self) -> None:
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        self.qf1_optimizer = torch.optim.Adam(self.qf1.parameters(), lr=self.lr)
        self.qf2_optimizer = torch.optim.Adam(self.qf2.parameters(), lr=self.lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def copy_from(self, source: "GoalSACAgent", reset_optimizers: bool = True, copy_update_count: bool = False) -> None:
        if self.obs_dim != source.obs_dim or self.action_dim != source.action_dim:
            raise ValueError("Cannot copy SAC agents with different observation/action dimensions.")
        self.policy.load_state_dict(source.policy.state_dict())
        self.qf1.load_state_dict(source.qf1.state_dict())
        self.qf2.load_state_dict(source.qf2.state_dict())
        self.target_qf1.load_state_dict(source.target_qf1.state_dict())
        self.target_qf2.load_state_dict(source.target_qf2.state_dict())
        with torch.no_grad():
            self.log_alpha.copy_(source.log_alpha.detach())
        if reset_optimizers:
            self._reset_optimizers()
        if copy_update_count:
            self.total_updates = int(source.total_updates)
        else:
            self.total_updates = 0
        self.last_diagnostics = {}

    def load_state_dict_payload(self, payload: Dict[str, object], load_optimizers: bool = True) -> None:
        self.policy.load_state_dict(payload["policy"])
        self.qf1.load_state_dict(payload["qf1"])
        self.qf2.load_state_dict(payload["qf2"])
        self.target_qf1.load_state_dict(payload["target_qf1"])
        self.target_qf2.load_state_dict(payload["target_qf2"])
        with torch.no_grad():
            self.log_alpha.copy_(torch.as_tensor(payload["log_alpha"], dtype=torch.float32, device=self.device))
        self.total_updates = int(payload.get("total_updates", 0))
        if load_optimizers:
            self.policy_optimizer.load_state_dict(payload["policy_optimizer"])
            self.qf1_optimizer.load_state_dict(payload["qf1_optimizer"])
            self.qf2_optimizer.load_state_dict(payload["qf2_optimizer"])
            self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])

    def hyperparameters(self) -> Dict[str, object]:
        return {
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "lr": self.lr,
            "gamma": self.gamma,
            "tau": self.tau,
            "alpha_init": self.alpha_init,
            "alpha": self.alpha,
            "target_entropy_internal": self.target_entropy,
            "target_entropy_paper": abs(self.target_entropy),
            "grad_clip_value": self.grad_clip_value,
            "optimizer": "Adam",
        }


class FakeGoalHardware:
    def __init__(self, start_z_cm: float = -10.0, seed: int = 12345):
        self.start_z_cm = float(start_z_cm)
        self.rng = np.random.default_rng(seed)
        self.pose_cm = np.asarray([0.0, 0.0, self.start_z_cm], dtype=np.float32)
        self.orientation_rad = np.zeros(3, dtype=np.float32)
        self.motor_angles_deg = np.zeros(MOTOR_COUNT, dtype=np.float32)
        self.step_count = 0

    def reset(self) -> None:
        self.pose_cm = np.asarray([0.0, 0.0, self.start_z_cm], dtype=np.float32)
        self.orientation_rad = np.zeros(3, dtype=np.float32)
        self.motor_angles_deg = np.zeros(MOTOR_COUNT, dtype=np.float32)
        self.step_count = 0

    def read_sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.pose_cm.copy(), self.orientation_rad.copy(), self.motor_angles_deg.copy()

    def write_action(self, action: Sequence[float]) -> None:
        action_array = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.motor_angles_deg = action_array * 55.0
        self.step_count += 1
        wave = float(np.mean(np.abs(action_array)))
        signed_balance = float(np.mean(action_array[::2]) - np.mean(action_array[1::2]))
        self.pose_cm[2] -= 0.10 + 0.04 * wave
        self.pose_cm[0] += 0.01 * signed_balance
        self.orientation_rad[0] = 0.02 * math.sin(self.step_count / 5.0)
        self.orientation_rad[1] = 0.03 * signed_balance
        self.orientation_rad[2] = 0.02 * math.cos(self.step_count / 7.0)


class GoalSnakeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        target_z_cm: float = -85.0,
        dmax_cm: float = DEFAULT_DMAX_CM,
        max_episode_steps: int = 175,
        profile_velocity: int = 120,
        action_settle_s: float = 0.3,
        hardware_disabled: bool = False,
        interactive_reset: bool = True,
        success_bonus: float = 0.0,
        time_penalty: float = 0.0,
        seed: int = 12345,
    ):
        super().__init__()
        self.target_z_cm = float(target_z_cm)
        self.dmax_cm = float(dmax_cm)
        self.max_episode_steps = int(max_episode_steps)
        self.profile_velocity = int(profile_velocity)
        self.action_settle_s = float(action_settle_s)
        self.hardware_disabled = bool(hardware_disabled)
        self.interactive_reset = bool(interactive_reset)
        self.success_bonus = float(success_bonus)
        self.time_penalty = float(time_penalty)
        self.current_terrain = "carpet"
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32)
        self.observation_labels = list(OBSERVATION_LABELS)
        self.info_sizes = {}

        self._sample_lock = threading.Lock()
        self._last_position_cm = np.asarray([0.0, 0.0, -10.0], dtype=np.float32)
        self._last_orientation_rad = np.zeros(3, dtype=np.float32)
        self._last_motor_angles_deg = np.zeros(MOTOR_COUNT, dtype=np.float32)
        self._last_opti_update = 0.0
        self._last_motor_update = 0.0
        self._motor_io_lock = threading.Lock()

        self.start_position_cm = self._last_position_cm.copy()
        self.start_orientation_rad = self._last_orientation_rad.copy()
        self.step_count = 0
        self.fake_hardware = FakeGoalHardware(seed=seed) if self.hardware_disabled else None
        self.motors = None
        self.opti = None
        if not self.hardware_disabled:
            self._initialize_hardware()

    @property
    def target_position_cm(self) -> np.ndarray:
        return np.asarray(
            [
                float(self.start_position_cm[0]),
                float(self.start_position_cm[1]),
                float(self.target_z_cm),
            ],
            dtype=np.float32,
        )

    def _initialize_hardware(self) -> None:
        os.environ["SNAKE_DXL_PROFILE_VELOCITY"] = str(self.profile_velocity)
        import motorssynced
        import optitrack

        self.motors = motorssynced.MotorsSynced()
        self.opti = optitrack.Optitrack()
        min_count, max_count = self.goal_motor_count_bounds()
        self.motors.MIN_POS = float(min_count)
        self.motors.MAX_POS = float(max_count)
        self.motors.PROFILE_VELOCITY = int(self.profile_velocity)
        self.motors.setMotorSpeed()

    @staticmethod
    def goal_motor_count_bounds() -> Tuple[int, int]:
        counts = action_to_motor_counts(np.asarray([-1.0] * MOTOR_COUNT, dtype=np.float32))
        min_count = int(counts[0])
        counts = action_to_motor_counts(np.asarray([1.0] * MOTOR_COUNT, dtype=np.float32))
        max_count = int(counts[0])
        return min_count, max_count

    def start_background_threads(self, stop_event: threading.Event) -> List[threading.Thread]:
        if self.hardware_disabled:
            return []
        threads = [
            threading.Thread(target=self._opti_loop, args=(stop_event,), daemon=True),
            threading.Thread(target=self._motor_loop, args=(stop_event,), daemon=True),
        ]
        for thread in threads:
            thread.start()
        return threads

    def _opti_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self._poll_opti_once()
            except Exception as exc:
                print(f"OptiTrack polling warning: {exc}")
            time.sleep(0.001)

    def _motor_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            try:
                self._poll_motor_once()
            except Exception as exc:
                print(f"Motor polling warning: {exc}")
            time.sleep(0.02)

    def _poll_opti_once(self) -> None:
        if self.opti is None:
            return
        position_m, orientation_deg = self.opti.optiTrackGetPos()
        position_cm = np.asarray(position_m, dtype=np.float32).reshape(3) * 100.0
        orientation_rad = optitrack_xzy_degrees_to_physical_xyz_radians(orientation_deg)
        with self._sample_lock:
            self._last_position_cm = position_cm
            self._last_orientation_rad = orientation_rad
            self._last_opti_update = time.time()

    def _poll_motor_once(self) -> None:
        if self.motors is None:
            return
        with self._motor_io_lock:
            motor_norm = np.asarray(self.motors.readPos(recover_on_failure=False), dtype=np.float32).reshape(-1)
        if motor_norm.size != MOTOR_COUNT:
            return
        if np.max(np.abs(motor_norm)) <= 1.5:
            motor_angles = np.clip(motor_norm, -1.0, 1.0) * 55.0
        else:
            motor_angles = motor_counts_to_degrees(motor_norm)
        with self._sample_lock:
            self._last_motor_angles_deg = motor_angles.astype(np.float32)
            self._last_motor_update = time.time()

    def _read_sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.hardware_disabled:
            return self.fake_hardware.read_sample()
        self._poll_opti_once()
        self._poll_motor_once()
        with self._sample_lock:
            return (
                self._last_position_cm.copy(),
                self._last_orientation_rad.copy(),
                self._last_motor_angles_deg.copy(),
            )

    def _write_action(self, action: Sequence[float]) -> np.ndarray:
        action_array = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        motor_counts = action_to_motor_counts(action_array)
        if self.hardware_disabled:
            self.fake_hardware.write_action(action_array)
            return motor_counts
        if self.motors is None:
            raise RuntimeError("Motor hardware is not initialized.")
        with self._motor_io_lock:
            write_ok = self.motors.writePos([int(value) for value in motor_counts])
        if not write_ok:
            raise RuntimeError(f"Failed to write GOAL-limited motor command {motor_counts.tolist()}.")
        time.sleep(self.action_settle_s)
        return motor_counts

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        prompt = bool(options.get("interactive_reset", self.interactive_reset))
        self.step_count = 0
        if self.hardware_disabled:
            self.fake_hardware.reset()
        else:
            if prompt and self.motors is not None:
                with self._motor_io_lock:
                    self.motors.disableTorque()
                try:
                    input("Reset robot by hand to the shared start pose, then press Enter.")
                except EOFError:
                    print("Reset prompt skipped because stdin is not interactive.")
                self._poll_motor_once()
                with self._motor_io_lock:
                    self.motors.enableTorque()
            elif self.motors is not None:
                self._poll_motor_once()

        position_cm, orientation_rad, motor_angles_deg = self._read_sample()
        self.start_position_cm = position_cm.copy()
        self.start_orientation_rad = orientation_rad.copy()
        observation, raw = build_goal_observation(
            self.start_position_cm,
            position_cm,
            self.start_orientation_rad,
            orientation_rad,
            motor_angles_deg,
            target_position_cm=self.target_position_cm,
            use_target_relative_position=True,
            dmax=self.dmax_cm,
        )
        info = self._info_payload(position_cm, orientation_rad, motor_angles_deg, raw)
        return observation, info

    def step(self, action):
        action_array = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        motor_counts = self._write_action(action_array)
        position_cm, orientation_rad, motor_angles_deg = self._read_sample()
        self.step_count += 1
        observation, raw = build_goal_observation(
            self.start_position_cm,
            position_cm,
            self.start_orientation_rad,
            orientation_rad,
            motor_angles_deg,
            target_position_cm=self.target_position_cm,
            use_target_relative_position=True,
            dmax=self.dmax_cm,
        )
        if self.target_z_cm < float(self.start_position_cm[2]):
            terminated = float(position_cm[2]) <= self.target_z_cm
        else:
            terminated = float(position_cm[2]) >= self.target_z_cm
        truncated = self.step_count >= self.max_episode_steps
        reward, reward_components = compute_goal_reward(
            start_z_cm=float(self.start_position_cm[2]),
            current_z_cm=float(position_cm[2]),
            target_z_cm=self.target_z_cm,
            yaw_rad=raw["theta_y_rad"],
            dmax=self.dmax_cm,
            success_bonus=self.success_bonus,
            time_penalty=self.time_penalty,
            terminated_by_target=bool(terminated),
        )
        info = self._info_payload(position_cm, orientation_rad, motor_angles_deg, raw)
        info.update(reward_components)
        info.update(
            {
                "terrain": self.current_terrain,
                "step": self.step_count,
                "motor_command_counts": motor_counts.tolist(),
                "normalized_action": action_array.tolist(),
                "target_z_cm": self.target_z_cm,
                "yaw_penalty_axis": "y",
                "terminated_by_target": bool(terminated),
                "truncated_by_episode_length": bool(truncated and not terminated),
            }
        )
        return observation, reward, bool(terminated), bool(truncated and not terminated), info

    def _info_payload(
        self,
        position_cm: np.ndarray,
        orientation_rad: np.ndarray,
        motor_angles_deg: np.ndarray,
        raw_observation: Dict[str, float],
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "position_x_cm": float(position_cm[0]),
            "position_y_cm": float(position_cm[1]),
            "position_z_cm": float(position_cm[2]),
            "orientation_x_rad": float(orientation_rad[0]),
            "orientation_y_rad": float(orientation_rad[1]),
            "orientation_z_rad": float(orientation_rad[2]),
            "start_x_cm": float(self.start_position_cm[0]),
            "start_y_cm": float(self.start_position_cm[1]),
            "start_z_cm": float(self.start_position_cm[2]),
            "dmax_cm": float(self.dmax_cm),
        }
        payload.update(raw_observation)
        for idx, value in enumerate(motor_angles_deg, start=1):
            payload[f"motor{idx}_angle_deg"] = float(value)
        return payload

    def close(self) -> None:
        if self.hardware_disabled or self.motors is None:
            return
        try:
            with self._motor_io_lock:
                self.motors.disableTorque()
        except Exception as exc:
            print(f"Motor shutdown warning: {exc}")


def append_csv_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new_fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in new_fieldnames:
                new_fieldnames.append(key)
    existing_rows: List[Dict[str, str]] = []
    existing_fieldnames: List[str] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)
    fieldnames = list(existing_fieldnames)
    for key in new_fieldnames:
        if key not in fieldnames:
            fieldnames.append(key)
    if existing_rows and fieldnames != existing_fieldnames:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def robust_score_from_terrain_returns(terrain_returns: Dict[str, Sequence[float]]) -> float:
    means = [float(np.mean(values)) for values in terrain_returns.values() if len(values) > 0]
    if not means:
        return float("nan")
    return float(np.mean(means) - 0.5 * np.std(means))


def load_torch_payload(path: Path) -> Dict[str, object]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def write_json_atomic(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


class GoalResearchRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        set_global_seeds(int(args.seed))
        self.resume_requested = bool(getattr(args, "resume_run_dir", None))
        if self.resume_requested:
            self.output_dir = Path(args.resume_run_dir).resolve()
        else:
            self.output_dir = Path(args.output_dir).resolve() / self._run_name(args)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.morphology = build_morphology_config(
            layout=args.layout,
            a_width=args.a_width,
            a_angle=args.a_angle,
            b_width=args.b_width,
            b_angle=args.b_angle,
        )
        self.env = GoalSnakeEnv(
            target_z_cm=args.target_z,
            dmax_cm=args.dmax,
            max_episode_steps=args.episode_length,
            profile_velocity=args.profile_velocity,
            action_settle_s=args.action_settle_s,
            hardware_disabled=args.dry_run or args.hardware_disabled,
            interactive_reset=args.interactive_reset,
            success_bonus=getattr(args, "success_bonus", 0.0),
            time_penalty=getattr(args, "time_penalty", 0.0),
            seed=args.seed,
        )
        population_replay_size = int(getattr(args, "population_replay_size", getattr(args, "replay_size", DEFAULT_REPLAY_SIZE)))
        individual_replay_size = int(getattr(args, "individual_replay_size", getattr(args, "replay_size", DEFAULT_REPLAY_SIZE)))
        self.population_agent = GoalSACAgent(
            obs_dim=CONDITIONED_OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            hidden_sizes=HIDDEN_SIZES,
            lr=args.learning_rate,
            gamma=args.gamma,
            tau=args.tau,
            alpha_init=args.alpha_init,
            target_entropy=args.target_entropy,
            grad_clip_value=args.grad_clip_value,
        )
        self.individual_agent = GoalSACAgent(
            obs_dim=CONDITIONED_OBSERVATION_DIM,
            action_dim=ACTION_DIM,
            hidden_sizes=HIDDEN_SIZES,
            lr=args.learning_rate,
            gamma=args.gamma,
            tau=args.tau,
            alpha_init=args.alpha_init,
            target_entropy=args.target_entropy,
            grad_clip_value=args.grad_clip_value,
        )
        self.individual_agent.copy_from(self.population_agent, reset_optimizers=True)
        self.population_replay = GoalReplayBuffer(population_replay_size)
        self.individual_replay = GoalReplayBuffer(individual_replay_size)
        self.agent = self.individual_agent
        self.replay = self.individual_replay
        self.design_id = int(getattr(args, "design_id", 0))
        self.active_condition = build_condition_vector(self.morphology, self._terrain_order()[0])
        self.trained_individual_payloads: Dict[str, Dict[str, object]] = {}
        self.stop_event = threading.Event()
        self.background_threads: List[threading.Thread] = []
        self.step_csv = self.output_dir / "training_steps.csv"
        self.episode_csv = self.output_dir / "episode_summary.csv"
        self.loss_csv = self.output_dir / "losses.csv"
        self.eval_step_csv = self.output_dir / "eval_steps.csv"
        self.eval_summary_csv = self.output_dir / "eval_summary.csv"
        self.metadata_json = self.output_dir / "run_metadata.json"
        self.summary_json = self.output_dir / "final_summary.json"
        self.resume_json = self.output_dir / "resume_state.json"
        self.completed_training_episodes = 0
        self.resume_state: Dict[str, object] = {}
        self._save_metadata()
        if self.resume_requested:
            self._load_resume_state()

    @staticmethod
    def _run_name(args: argparse.Namespace) -> str:
        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        config_id = args.config_id or f"{args.layout}_A{args.a_width:.3f}_{args.a_angle:.1f}"
        return f"{timestamp}_{config_id}"

    @property
    def individual_updates_per_episode(self) -> int:
        value = getattr(self.args, "individual_updates_per_episode", None)
        return int(self.args.updates_per_episode if value is None else value)

    @property
    def population_updates_per_episode(self) -> int:
        value = getattr(self.args, "population_updates_per_episode", None)
        return int(self.args.updates_per_episode if value is None else value)

    def _save_metadata(self) -> None:
        created_at = datetime.now().isoformat(timespec="seconds")
        if self.metadata_json.exists():
            try:
                created_at = json.loads(self.metadata_json.read_text(encoding="utf-8")).get("created_at", created_at)
            except Exception:
                pass
        metadata = {
            "created_at": created_at,
            "last_opened_at": datetime.now().isoformat(timespec="seconds"),
            "goal_pipeline": {
                "episodes_per_terrain": self.args.episodes_per_terrain,
                "updates_per_episode": self.args.updates_per_episode,
                "individual_updates_per_episode": self.individual_updates_per_episode,
                "population_updates_per_episode": self.population_updates_per_episode,
                "warmup_episodes_per_terrain": int(getattr(self.args, "warmup_episodes_per_terrain", 0)),
                "update_schedule": str(getattr(self.args, "update_schedule", "replay-ramp")),
                "update_ramp_min": int(getattr(self.args, "update_ramp_min", 50)),
                "update_ramp_replay_divisor": int(getattr(self.args, "update_ramp_replay_divisor", 4)),
                "episode_length": self.args.episode_length,
                "batch_size": self.args.batch_size,
                "target_z_cm": self.args.target_z,
                "dmax_cm": self.args.dmax,
                "profile_velocity": self.args.profile_velocity,
                "cooldown_seconds": 0,
                "rigid_body_id": 99,
                "observation_labels": OBSERVATION_LABELS,
                "condition_labels": CONDITION_LABELS,
                "conditioned_observation_dim": CONDITIONED_OBSERVATION_DIM,
                "position_observation_mode": "target_relative",
                "yaw_penalty_axis": "y",
                "success_bonus": float(getattr(self.args, "success_bonus", 0.0)),
                "time_penalty": float(getattr(self.args, "time_penalty", 0.0)),
                "action_limit_deg": 55.0,
            },
            "morphology": self.morphology.metadata(),
            "sac": {
                "population": self.population_agent.hyperparameters(),
                "individual": self.individual_agent.hyperparameters(),
                "copy_direction": "population_to_individual_at_each_design_terrain_block",
                "individual_to_population_transfer": "replay_data_only",
            },
            "replay": {
                "population_replay_size": self.population_replay.capacity,
                "individual_replay_size": self.individual_replay.capacity,
            },
            "terrain_order": self._terrain_order(),
            "dry_run": bool(self.args.dry_run or self.args.hardware_disabled),
            "resume": {
                "enabled": True,
                "resume_manifest": str(self.resume_json),
                "resume_requested": bool(self.resume_requested),
            },
        }
        write_json_atomic(self.metadata_json, metadata)

    def _terrain_order(self) -> List[str]:
        return [terrain.strip() for terrain in self.args.terrain_order.split(",") if terrain.strip()]

    def run(self) -> Dict[str, object]:
        self._print_morphology_prompt()
        if not (self.args.dry_run or self.args.hardware_disabled):
            self.background_threads = self.env.start_background_threads(self.stop_event)
        try:
            completed_before_run = int(self.completed_training_episodes)
            episode_index = 0
            for terrain in self._terrain_order():
                block_start_episode = episode_index + 1
                block_end_episode = episode_index + int(self.args.episodes_per_terrain)
                if completed_before_run >= block_end_episode:
                    episode_index = block_end_episode
                    continue
                self._prompt_for_terrain(terrain)
                if completed_before_run < block_start_episode:
                    self.start_design_terrain_block(terrain)
                else:
                    self.active_condition = self.condition_for(terrain)
                    self.agent = self.individual_agent
                    self.replay = self.individual_replay

                for episode_in_terrain in range(1, int(self.args.episodes_per_terrain) + 1):
                    episode_index += 1
                    if episode_index <= completed_before_run:
                        continue
                    summary = self.collect_episode(
                        terrain=terrain,
                        episode_index=episode_index,
                        episode_in_terrain=episode_in_terrain,
                        deterministic=False,
                        add_to_replay=True,
                        csv_path=self.step_csv,
                        phase="train",
                    )
                    append_csv_rows(self.episode_csv, [summary])
                    loss_rows = self._run_updates(episode_index, terrain, episode_in_terrain)
                    append_csv_rows(self.loss_csv, loss_rows)
                    terrain_completed = episode_in_terrain == int(self.args.episodes_per_terrain)
                    if terrain_completed:
                        self.trained_individual_payloads[terrain] = self.individual_agent.state_dict()
                    replay_paths = self._save_replay_snapshots()
                    checkpoint_path = self._save_checkpoint(
                        episode_index,
                        terrain,
                        episode_in_terrain=episode_in_terrain,
                        terrain_completed=terrain_completed,
                        population_replay_path=replay_paths[0],
                        individual_replay_path=replay_paths[1],
                    )
                    self._save_resume_state(
                        checkpoint_path=checkpoint_path,
                        episode_index=episode_index,
                        terrain=terrain,
                        episode_in_terrain=episode_in_terrain,
                        terrain_completed=terrain_completed,
                        population_replay_path=replay_paths[0],
                        individual_replay_path=replay_paths[1],
                    )
                    self.completed_training_episodes = episode_index
                    print(
                        f"Episode {episode_index} {terrain} done: return={summary['episode_return']:.4f}, "
                        f"individual_replay={len(self.individual_replay)}, "
                        f"population_replay={len(self.population_replay)}, updates={len(loss_rows)}"
                    )
                self.trained_individual_payloads[terrain] = self.individual_agent.state_dict()

            eval_summary = self.evaluate()
            self.population_replay.save_npz(self.output_dir / "population_replay.npz")
            self.summary_json.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
            return eval_summary
        finally:
            self.stop_event.set()
            for thread in self.background_threads:
                thread.join(timeout=2.0)
            self.env.close()

    def _print_morphology_prompt(self) -> None:
        print("Install/verify this scale configuration before starting:")
        print(json.dumps(self.morphology.metadata(), indent=2))
        if self.args.interactive_reset and not (self.args.dry_run or self.args.hardware_disabled):
            print("The robot will prompt for a manual reset before every episode.")

    def _prompt_for_terrain(self, terrain: str) -> None:
        if self.args.dry_run or self.args.hardware_disabled:
            return
        if not bool(getattr(self.args, "terrain_change_prompt", True)):
            return
        try:
            input(f"Install/verify terrain '{terrain}', then press Enter.")
        except EOFError:
            print(f"Terrain prompt skipped because stdin is not interactive. Expected terrain: {terrain}.")

    def condition_for(self, terrain: str, morphology: Optional[MorphologyConfig] = None) -> np.ndarray:
        return build_condition_vector(morphology or self.morphology, terrain)

    def start_design_terrain_block(self, terrain: str) -> None:
        self.active_condition = self.condition_for(terrain)
        self.individual_agent.copy_from(self.population_agent, reset_optimizers=True)
        self.individual_replay.clear()
        self.agent = self.individual_agent
        self.replay = self.individual_replay

    def _checkpoint_run_data(
        self,
        episode_index: int,
        terrain: str,
        episode_in_terrain: int,
        terrain_completed: bool,
        population_replay_path: Optional[Path] = None,
        individual_replay_path: Optional[Path] = None,
    ) -> Dict[str, object]:
        return {
            "run_id": self.output_dir.name,
            "output_dir": str(self.output_dir),
            "config_id": self.args.config_id,
            "design_id": self.design_id,
            "layout": self.morphology.layout,
            "morphology": self.morphology.metadata(),
            "design_vector": self.morphology.as_design_vector(),
            "terrain": terrain,
            "terrain_order": self._terrain_order(),
            "episode": int(episode_index),
            "episode_in_terrain": int(episode_in_terrain),
            "episodes_per_terrain": int(self.args.episodes_per_terrain),
            "terrain_completed": bool(terrain_completed),
            "next_episode": int(episode_index) + 1,
            "target_z_cm": float(self.args.target_z),
            "dmax_cm": float(self.args.dmax),
            "episode_length": int(self.args.episode_length),
            "batch_size": int(self.args.batch_size),
            "profile_velocity": int(self.args.profile_velocity),
            "yaw_penalty_axis": "y",
            "success_bonus": float(getattr(self.args, "success_bonus", 0.0)),
            "time_penalty": float(getattr(self.args, "time_penalty", 0.0)),
            "action_limit_deg": 55.0,
            "condition_labels": CONDITION_LABELS,
            "condition": self.condition_for(terrain).tolist(),
            "population_replay": str(population_replay_path) if population_replay_path else "",
            "individual_replay": str(individual_replay_path) if individual_replay_path else "",
            "resume_manifest": str(self.resume_json),
        }

    def _save_replay_snapshots(self) -> Tuple[Path, Path]:
        replay_dir = self.output_dir / "replay"
        population_path = replay_dir / "population_replay_latest.npz"
        individual_path = replay_dir / "individual_replay_latest.npz"
        self.population_replay.save_npz(population_path)
        self.individual_replay.save_npz(individual_path)
        return population_path, individual_path

    def _save_resume_state(
        self,
        checkpoint_path: Path,
        episode_index: int,
        terrain: str,
        episode_in_terrain: int,
        terrain_completed: bool,
        population_replay_path: Path,
        individual_replay_path: Path,
    ) -> None:
        payload = self._checkpoint_run_data(
            episode_index=episode_index,
            terrain=terrain,
            episode_in_terrain=episode_in_terrain,
            terrain_completed=terrain_completed,
            population_replay_path=population_replay_path,
            individual_replay_path=individual_replay_path,
        )
        payload.update(
            {
                "latest_completed_episode": int(episode_index),
                "latest_checkpoint": str(checkpoint_path),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "completed_terrain_policies": sorted(self.trained_individual_payloads.keys()),
            }
        )
        write_json_atomic(self.resume_json, payload)

    def _latest_checkpoint_from_dir(self) -> Optional[Path]:
        checkpoint_dir = self.output_dir / "checkpoints"
        if not checkpoint_dir.exists():
            return None
        checkpoints = sorted(checkpoint_dir.glob("episode_*_*.pt"))
        return checkpoints[-1] if checkpoints else None

    def _load_resume_state(self) -> None:
        if self.resume_json.exists():
            self.resume_state = json.loads(self.resume_json.read_text(encoding="utf-8"))
            checkpoint_path = Path(str(self.resume_state.get("latest_checkpoint", "")))
            population_replay_path = Path(str(self.resume_state.get("population_replay", "")))
            individual_replay_path = Path(str(self.resume_state.get("individual_replay", "")))
        else:
            checkpoint_path = self._latest_checkpoint_from_dir()
            population_replay_path = self.output_dir / "replay" / "population_replay_latest.npz"
            individual_replay_path = self.output_dir / "replay" / "individual_replay_latest.npz"
            self.resume_state = {}

        if checkpoint_path is None or not checkpoint_path.exists():
            print(f"No resume checkpoint found in {self.output_dir}; starting from episode 1.")
            return

        checkpoint = load_torch_payload(checkpoint_path)
        checkpoint_morphology = checkpoint.get("morphology", {})
        checkpoint_layout = str(checkpoint.get("layout", checkpoint_morphology.get("layout", "")))
        checkpoint_design_vector = checkpoint.get("design_vector", checkpoint_morphology.get("design_vector", []))
        if checkpoint_layout and checkpoint_layout != self.morphology.layout:
            raise ValueError(
                f"Resume checkpoint layout '{checkpoint_layout}' does not match requested layout "
                f"'{self.morphology.layout}'."
            )
        if checkpoint_design_vector and not np.allclose(
            np.asarray(checkpoint_design_vector, dtype=np.float64),
            np.asarray(self.morphology.as_design_vector(), dtype=np.float64),
        ):
            raise ValueError("Resume checkpoint design vector does not match requested morphology.")
        self.population_agent.load_state_dict_payload(checkpoint["population_agent"], load_optimizers=True)
        self.individual_agent.load_state_dict_payload(checkpoint["individual_agent"], load_optimizers=True)
        self.trained_individual_payloads = checkpoint.get("trained_individual_payloads", {})
        self.completed_training_episodes = int(checkpoint.get("episode", self.resume_state.get("latest_completed_episode", 0)))
        if population_replay_path.exists():
            self.population_replay.load_npz(population_replay_path)
        else:
            print(f"Resume warning: population replay snapshot missing at {population_replay_path}.")
        if individual_replay_path.exists():
            self.individual_replay.load_npz(individual_replay_path)
        else:
            print(f"Resume warning: individual replay snapshot missing at {individual_replay_path}.")
        terrain = str(checkpoint.get("terrain", self.resume_state.get("terrain", self._terrain_order()[0])))
        self.active_condition = self.condition_for(terrain)
        self.agent = self.individual_agent
        self.replay = self.individual_replay
        print(
            f"Resumed {self.output_dir.name} from completed episode {self.completed_training_episodes} "
            f"({terrain})."
        )

    def collect_episode(
        self,
        terrain: str,
        episode_index: int,
        episode_in_terrain: int,
        deterministic: bool,
        add_to_replay: bool,
        csv_path: Path,
        phase: str,
        policy_agent: Optional[GoalSACAgent] = None,
        policy_role: str = "individual",
    ) -> Dict[str, object]:
        active_policy = policy_agent or self.individual_agent
        self.env.current_terrain = terrain
        robot_observation, reset_info = self.env.reset(
            options={"interactive_reset": bool(self.args.interactive_reset and not self.args.dry_run)}
        )
        condition = self.condition_for(terrain)
        observation = build_conditioned_observation(robot_observation, condition)
        terrain_id = TERRAIN_LABELS.index(terrain) if terrain in TERRAIN_LABELS else -1
        rows = []
        episode_return = 0.0
        terminal = False
        final_info = dict(reset_info)
        for step_idx in range(1, int(self.args.episode_length) + 1):
            action = active_policy.act(observation, deterministic=deterministic)
            next_robot_observation, reward, terminated, truncated, info = self.env.step(action)
            next_observation = build_conditioned_observation(next_robot_observation, condition)
            done = bool(terminated or truncated)
            terminal_for_replay = bool(terminated)
            if add_to_replay:
                self.individual_replay.add_sample(
                    robot_observation,
                    action,
                    reward,
                    next_robot_observation,
                    terminal_for_replay,
                    condition=condition,
                    next_condition=condition,
                    terrain_id=terrain_id,
                    design_id=self.design_id,
                )
                self.population_replay.add_sample(
                    robot_observation,
                    action,
                    reward,
                    next_robot_observation,
                    terminal_for_replay,
                    condition=condition,
                    next_condition=condition,
                    terrain_id=terrain_id,
                    design_id=self.design_id,
                )
            row = self._step_row(
                phase=phase,
                terrain=terrain,
                episode_index=episode_index,
                    episode_in_terrain=episode_in_terrain,
                    step_idx=step_idx,
                    policy_role=policy_role,
                    robot_observation=robot_observation,
                condition=condition,
                action=action,
                reward=reward,
                info=info,
            )
            rows.append(row)
            if bool(getattr(self.args, "print_step_rewards", True)):
                self._print_step_reward(
                    phase=phase,
                    terrain=terrain,
                    episode_index=episode_index,
                    step_idx=step_idx,
                    reward=reward,
                    info=info,
                )
            episode_return += float(reward)
            robot_observation = next_robot_observation
            observation = next_observation
            terminal = done
            final_info = dict(info)
            if done:
                break
        append_csv_rows(csv_path, rows)
        return {
            "phase": phase,
            "policy_role": policy_role,
            "terrain": terrain,
            "episode": episode_index,
            "episode_in_terrain": episode_in_terrain,
            "steps": len(rows),
            "episode_return": float(episode_return),
            "ended": bool(terminal),
            "terminated_by_target": bool(final_info.get("terminated_by_target", False)),
            "truncated_by_episode_length": bool(final_info.get("truncated_by_episode_length", False)),
            "success": bool(final_info.get("terminated_by_target", False)),
            "remaining_distance_cm": float(final_info.get("delta_forward_cm", 0.0)),
            "forward_progress_cm": float(final_info.get("forward_curr_cm", 0.0)),
            "final_z_cm": float(final_info.get("position_z_cm", np.nan)),
            "target_z_cm": float(self.args.target_z),
            "individual_alpha": self.individual_agent.alpha,
            "population_alpha": self.population_agent.alpha,
            "individual_replay_size": len(self.individual_replay),
            "population_replay_size": len(self.population_replay),
            "layout": self.morphology.layout,
            "A_width_ratio": self.morphology.a_width,
            "A_attack_angle_deg": self.morphology.a_angle,
            "B_width_ratio": self.morphology.b_width,
            "B_attack_angle_deg": self.morphology.b_angle,
        }

    def _print_step_reward(
        self,
        phase: str,
        terrain: str,
        episode_index: int,
        step_idx: int,
        reward: float,
        info: Dict[str, object],
    ) -> None:
        if phase != "train":
            return
        print(
            f"{terrain} ep={episode_index:03d} step={step_idx:03d} "
            f"reward={float(reward):+.4f} "
            f"forward={float(info.get('forward_reward', np.nan)):+.4f} "
            f"yaw_penalty_rad={float(info.get('yaw_penalty_rad', np.nan)):.4f} "
            f"alpha={self.individual_agent.alpha:.5f}"
        )

    def _step_row(
        self,
        phase: str,
        terrain: str,
        episode_index: int,
        episode_in_terrain: int,
        step_idx: int,
        policy_role: str,
        robot_observation: np.ndarray,
        condition: np.ndarray,
        action: np.ndarray,
        reward: float,
        info: Dict[str, object],
    ) -> Dict[str, object]:
        row: Dict[str, object] = {
            "phase": phase,
            "terrain": terrain,
            "policy_role": policy_role,
            "episode": episode_index,
            "episode_in_terrain": episode_in_terrain,
            "step": step_idx,
            "reward": float(reward),
            "individual_alpha": self.individual_agent.alpha,
            "population_alpha": self.population_agent.alpha,
            "layout": self.morphology.layout,
            "A_width_ratio": self.morphology.a_width,
            "A_attack_angle_deg": self.morphology.a_angle,
            "B_width_ratio": self.morphology.b_width,
            "B_attack_angle_deg": self.morphology.b_angle,
        }
        for idx, value in enumerate(robot_observation, start=1):
            label = OBSERVATION_LABELS[idx - 1]
            row[label] = float(value)
        for idx, value in enumerate(condition, start=1):
            label = CONDITION_LABELS[idx - 1]
            row[label] = float(value)
        for idx, value in enumerate(action, start=1):
            row[f"Motor{idx}_Action"] = float(value)
        for key, value in info.items():
            if isinstance(value, (int, float, str, bool)):
                row[key] = value
        return row

    def _run_updates(self, episode_index: int, terrain: str, episode_in_terrain: int) -> List[Dict[str, object]]:
        rows = []
        warmup_episodes = int(getattr(self.args, "warmup_episodes_per_terrain", 0))
        if int(episode_in_terrain) <= warmup_episodes:
            print(
                f"Skipping SAC updates for warmup episode {episode_in_terrain}/{warmup_episodes} "
                f"on {terrain}."
            )
            return rows
        if len(self.individual_replay) > 0:
            individual_update_count = self._scheduled_update_count(
                replay=self.individual_replay,
                configured_max=self.individual_updates_per_episode,
            )
            rows.extend(
                self._run_agent_updates(
                    role="individual",
                    agent=self.individual_agent,
                    replay=self.individual_replay,
                    update_count=individual_update_count,
                    episode_index=episode_index,
                    terrain=terrain,
                )
            )
        if len(self.population_replay) > 0:
            population_update_count = self._scheduled_update_count(
                replay=self.population_replay,
                configured_max=self.population_updates_per_episode,
            )
            rows.extend(
                self._run_agent_updates(
                    role="population",
                    agent=self.population_agent,
                    replay=self.population_replay,
                    update_count=population_update_count,
                    episode_index=episode_index,
                    terrain=terrain,
                )
            )
        return rows

    def _scheduled_update_count(self, replay: GoalReplayBuffer, configured_max: int) -> int:
        if str(getattr(self.args, "update_schedule", "replay-ramp")) == "fixed":
            return int(configured_max)
        min_updates = int(getattr(self.args, "update_ramp_min", 50))
        divisor = max(int(getattr(self.args, "update_ramp_replay_divisor", 4)), 1)
        replay_scaled_updates = len(replay) // divisor
        return int(min(int(configured_max), max(min_updates, replay_scaled_updates)))

    def _run_agent_updates(
        self,
        role: str,
        agent: GoalSACAgent,
        replay: GoalReplayBuffer,
        update_count: int,
        episode_index: int,
        terrain: str,
    ) -> List[Dict[str, object]]:
        rows = []
        for update_idx in range(1, int(update_count) + 1):
            batch = replay.random_batch(int(self.args.batch_size))
            diagnostics = agent.train_step(batch)
            row = {
                "role": role,
                "episode": episode_index,
                "terrain": terrain,
                "update_in_episode": update_idx,
                "replay_size": len(replay),
            }
            row.update(diagnostics)
            rows.append(row)
        return rows

    def _save_checkpoint(
        self,
        episode_index: int,
        terrain: str,
        episode_in_terrain: int,
        terrain_completed: bool,
        population_replay_path: Path,
        individual_replay_path: Path,
    ) -> Path:
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"episode_{episode_index:03d}_{terrain}.pt"
        run_data = self._checkpoint_run_data(
            episode_index=episode_index,
            terrain=terrain,
            episode_in_terrain=episode_in_terrain,
            terrain_completed=terrain_completed,
            population_replay_path=population_replay_path,
            individual_replay_path=individual_replay_path,
        )
        torch.save(
            {
                "episode": episode_index,
                "terrain": terrain,
                "episode_in_terrain": int(episode_in_terrain),
                "terrain_completed": bool(terrain_completed),
                "run_data": run_data,
                "design_id": self.design_id,
                "layout": self.morphology.layout,
                "design_vector": self.morphology.as_design_vector(),
                "individual_agent": self.individual_agent.state_dict(),
                "population_agent": self.population_agent.state_dict(),
                "trained_individual_payloads": self.trained_individual_payloads,
                "morphology": self.morphology.metadata(),
                "observation_labels": OBSERVATION_LABELS,
                "condition_labels": CONDITION_LABELS,
            },
            path,
        )
        return path

    def evaluate(self) -> Dict[str, object]:
        individual_eval = self._evaluate_policy_pass(
            policy_role="individual_per_terrain",
            phase="eval",
            policy_agent=None,
            use_terrain_individual_payloads=True,
        )
        summary = {
            "morphology": self.morphology.metadata(),
            "terrain_returns": individual_eval["terrain_returns"],
            "terrain_progress_cm": individual_eval["terrain_progress_cm"],
            "terrain_success_rate": individual_eval["terrain_success_rate"],
            "mean_return_by_terrain": individual_eval["mean_return_by_terrain"],
            "mean_progress_by_terrain_cm": individual_eval["mean_progress_by_terrain_cm"],
            "robust_score": individual_eval["robust_score"],
            "eval_policy": "individual_per_terrain",
            "individual_alpha": self.individual_agent.alpha,
            "population_alpha": self.population_agent.alpha,
            "individual_total_updates": self.individual_agent.total_updates,
            "population_total_updates": self.population_agent.total_updates,
            "individual_replay_size": len(self.individual_replay),
            "population_replay_size": len(self.population_replay),
        }
        if bool(getattr(self.args, "eval_population_policy", True)):
            summary["population_eval"] = self._evaluate_policy_pass(
                policy_role="population_shared",
                phase="eval_population",
                policy_agent=self.population_agent,
                use_terrain_individual_payloads=False,
            )
        return summary

    def _evaluate_policy_pass(
        self,
        policy_role: str,
        phase: str,
        policy_agent: Optional[GoalSACAgent],
        use_terrain_individual_payloads: bool,
    ) -> Dict[str, object]:
        eval_rows = []
        terrain_returns: Dict[str, List[float]] = {}
        terrain_progress: Dict[str, List[float]] = {}
        terrain_success: Dict[str, List[float]] = {}
        episode_index = 0
        for terrain in self._terrain_order():
            active_policy = policy_agent
            if use_terrain_individual_payloads and terrain in self.trained_individual_payloads:
                self.individual_agent.load_state_dict_payload(
                    self.trained_individual_payloads[terrain],
                    load_optimizers=False,
                )
                active_policy = self.individual_agent
            if active_policy is None:
                active_policy = self.individual_agent
            self.active_condition = self.condition_for(terrain)
            terrain_returns[terrain] = []
            terrain_progress[terrain] = []
            terrain_success[terrain] = []
            for eval_idx in range(1, int(self.args.eval_episodes_per_terrain) + 1):
                episode_index += 1
                summary = self.collect_episode(
                    terrain=terrain,
                    episode_index=episode_index,
                    episode_in_terrain=eval_idx,
                    deterministic=True,
                    add_to_replay=False,
                    csv_path=self.eval_step_csv,
                    phase=phase,
                    policy_agent=active_policy,
                    policy_role=policy_role,
                )
                terrain_returns[terrain].append(float(summary["episode_return"]))
                terrain_progress[terrain].append(float(summary["forward_progress_cm"]))
                terrain_success[terrain].append(float(summary["success"]))
                eval_rows.append(summary)
        append_csv_rows(self.eval_summary_csv, eval_rows)
        robust_score = robust_score_from_terrain_returns(terrain_returns)
        return {
            "policy_role": policy_role,
            "terrain_returns": terrain_returns,
            "terrain_progress_cm": terrain_progress,
            "terrain_success_rate": {
                terrain: float(np.mean(values)) if values else float("nan")
                for terrain, values in terrain_success.items()
            },
            "mean_return_by_terrain": {
                terrain: float(np.mean(values)) if values else float("nan")
                for terrain, values in terrain_returns.items()
            },
            "mean_progress_by_terrain_cm": {
                terrain: float(np.mean(values)) if values else float("nan")
                for terrain, values in terrain_progress.items()
            },
            "robust_score": robust_score,
        }

    def score_candidate_with_population(
        self,
        morphology: MorphologyConfig,
        terrain: str,
        batch_size: int = 256,
    ) -> float:
        if len(self.population_replay) <= 0:
            raise RuntimeError("Population replay is empty; train before population-critic PSO scoring.")
        robot_observations = self.population_replay.sample_robot_observations(batch_size)
        condition = self.condition_for(terrain, morphology=morphology)
        observations = np.asarray(
            [build_conditioned_observation(robot_obs, condition) for robot_obs in robot_observations],
            dtype=np.float32,
        )
        actions = self.population_agent.act_batch(observations, deterministic=True)
        q1, q2 = self.population_agent.q_values(observations, actions)
        return float(np.minimum(q1, q2).mean())

    def robust_population_score(
        self,
        morphology: MorphologyConfig,
        batch_size: int = 256,
        mode: str = "robust",
        terrain: str = "carpet",
    ) -> Dict[str, float]:
        if mode == "terrain":
            score = self.score_candidate_with_population(morphology, terrain=terrain, batch_size=batch_size)
            return {f"q_{terrain}": score, "robust_q": score}
        terrain_scores = {
            terrain_name: self.score_candidate_with_population(morphology, terrain=terrain_name, batch_size=batch_size)
            for terrain_name in TERRAIN_LABELS
        }
        values = list(terrain_scores.values())
        robust_q = float(np.mean(values) - 0.5 * np.std(values))
        payload = {f"q_{terrain_name}": value for terrain_name, value in terrain_scores.items()}
        payload["robust_q"] = robust_q
        return payload

    def propose_pso_designs(self) -> Dict[str, object]:
        batch_size = int(getattr(self.args, "pso_batch_size", 256))
        mode = str(getattr(self.args, "pso_score_mode", "robust"))
        terrain = str(getattr(self.args, "pso_terrain", "carpet"))
        homogeneous = run_population_critic_pso(
            scorer=lambda morph: self.robust_population_score(morph, batch_size=batch_size, mode=mode, terrain=terrain),
            layout="homogeneous",
            particles=int(getattr(self.args, "homogeneous_particles", 24)),
            iterations=int(getattr(self.args, "iterations", 40)),
            top_k=int(getattr(self.args, "top_k", 4)),
            seed=int(getattr(self.args, "seed", 12345)),
        )
        heterogeneous = run_population_critic_pso(
            scorer=lambda morph: self.robust_population_score(morph, batch_size=batch_size, mode=mode, terrain=terrain),
            layout="heterogeneous_ab",
            particles=int(getattr(self.args, "heterogeneous_particles", 24)),
            iterations=int(getattr(self.args, "iterations", 40)),
            top_k=int(getattr(self.args, "pair_count", 3)),
            seed=int(getattr(self.args, "seed", 12345)) + 17,
        )
        return {
            "homogeneous_candidates": homogeneous,
            "heterogeneous_candidates": heterogeneous,
            "placement": "modules 1,3,5,7 = A; modules 2,4,6,8 = B",
            "score_source": "population_critic",
        }


def morphology_from_pso_position(layout: str, position: Sequence[float]) -> MorphologyConfig:
    if layout == "homogeneous":
        width, angle = position
        return build_morphology_config("homogeneous", float(width), float(angle))
    a_width, a_angle, b_width, b_angle = position
    return build_morphology_config(
        "heterogeneous_ab",
        float(a_width),
        float(a_angle),
        float(b_width),
        float(b_angle),
    )


def run_population_critic_pso(
    scorer,
    layout: str,
    particles: int = 24,
    iterations: int = 40,
    top_k: int = 4,
    seed: int = 12345,
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    if layout == "homogeneous":
        low = np.asarray([SCALE_WIDTH_BOUNDS[0], SCALE_ANGLE_BOUNDS[0]], dtype=np.float64)
        high = np.asarray([SCALE_WIDTH_BOUNDS[1], SCALE_ANGLE_BOUNDS[1]], dtype=np.float64)
    elif layout == "heterogeneous_ab":
        low = np.asarray(
            [SCALE_WIDTH_BOUNDS[0], SCALE_ANGLE_BOUNDS[0], SCALE_WIDTH_BOUNDS[0], SCALE_ANGLE_BOUNDS[0]],
            dtype=np.float64,
        )
        high = np.asarray(
            [SCALE_WIDTH_BOUNDS[1], SCALE_ANGLE_BOUNDS[1], SCALE_WIDTH_BOUNDS[1], SCALE_ANGLE_BOUNDS[1]],
            dtype=np.float64,
        )
    else:
        raise ValueError("layout must be 'homogeneous' or 'heterogeneous_ab'.")

    span = high - low
    positions = low + rng.random((particles, low.size)) * span
    velocities = rng.uniform(-0.25, 0.25, size=positions.shape) * span
    personal_best = positions.copy()
    personal_payloads = [scorer(morphology_from_pso_position(layout, pos)) for pos in positions]
    personal_scores = np.asarray([payload["robust_q"] for payload in personal_payloads], dtype=np.float64)
    global_best = personal_best[int(np.argmax(personal_scores))].copy()
    global_score = float(np.max(personal_scores))
    candidates: List[Dict[str, float]] = []

    for _ in range(int(iterations)):
        r1 = rng.random(positions.shape)
        r2 = rng.random(positions.shape)
        velocities = 0.65 * velocities + 1.4 * r1 * (personal_best - positions) + 1.4 * r2 * (global_best - positions)
        velocities = np.clip(velocities, -0.25 * span, 0.25 * span)
        positions = np.clip(positions + velocities, low, high)
        for particle_idx, pos in enumerate(positions):
            morphology = morphology_from_pso_position(layout, pos)
            payload = scorer(morphology)
            score = float(payload["robust_q"])
            if score > float(personal_scores[particle_idx]):
                personal_best[particle_idx] = pos
                personal_scores[particle_idx] = score
                personal_payloads[particle_idx] = payload
            if score > global_score:
                global_score = score
                global_best = pos.copy()
            row = morphology.metadata()
            row.update(payload)
            row["predicted_score"] = score
            candidates.append(row)

    for pos, score, payload in zip(personal_best, personal_scores, personal_payloads):
        row = morphology_from_pso_position(layout, pos).metadata()
        row.update(payload)
        row["predicted_score"] = float(score)
        candidates.append(row)

    candidates.sort(key=lambda item: item["predicted_score"], reverse=True)
    selected: List[Dict[str, float]] = []
    for candidate in candidates:
        vector = np.asarray(candidate["design_vector"], dtype=np.float64)
        if any(np.linalg.norm(vector - np.asarray(row["design_vector"], dtype=np.float64)) < 1e-3 for row in selected):
            continue
        selected.append(candidate)
        if len(selected) >= int(top_k):
            break
    return selected


def _summary_score(row: Dict[str, str]) -> float:
    for key in ("robust_score", "score", "eval_score"):
        if key in row and str(row[key]).strip() != "":
            return float(row[key])
    carpet = float(row.get("mean_return_carpet", row.get("carpet_return", "nan")))
    cardboard = float(row.get("mean_return_cardboard", row.get("cardboard_return", "nan")))
    values = [value for value in (carpet, cardboard) if np.isfinite(value)]
    if not values:
        return float("nan")
    return float(np.mean(values) - 0.5 * np.std(values))


def load_morphology_summary_csv(path: Path) -> List[Dict[str, float]]:
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            score = _summary_score(row)
            if not np.isfinite(score):
                continue
            rows.append(
                {
                    "width": float(row.get("A_width_ratio", row.get("width", row.get("a_width", 0.0)))),
                    "angle": float(row.get("A_attack_angle_deg", row.get("angle", row.get("a_angle", 0.0)))),
                    "score": float(score),
                    "layout": str(row.get("layout", "homogeneous")),
                }
            )
    return rows


def predict_design_score(width: float, angle: float, summaries: Sequence[Dict[str, float]]) -> float:
    if not summaries:
        center_width = 0.675
        center_angle = 15.0
        width_term = -abs(float(width) - center_width)
        angle_term = -abs(float(angle) - center_angle) / 30.0
        return float(width_term + angle_term)
    points = np.asarray([[row["width"], row["angle"]] for row in summaries], dtype=np.float64)
    scores = np.asarray([row["score"] for row in summaries], dtype=np.float64)
    scale = np.asarray([0.45, 30.0], dtype=np.float64)
    query = np.asarray([float(width), float(angle)], dtype=np.float64)
    dist = np.linalg.norm((points - query) / scale, axis=1)
    weights = 1.0 / np.maximum(dist, 1e-3) ** 2
    return float(np.sum(weights * scores) / np.sum(weights))


def run_pso_design_library(
    summaries: Sequence[Dict[str, float]],
    particles: int = 24,
    iterations: int = 40,
    top_k: int = 4,
    seed: int = 12345,
    width_bounds: Tuple[float, float] = (0.45, 0.90),
    angle_bounds: Tuple[float, float] = (0.0, 30.0),
) -> List[Dict[str, float]]:
    rng = np.random.default_rng(seed)
    low = np.asarray([width_bounds[0], angle_bounds[0]], dtype=np.float64)
    high = np.asarray([width_bounds[1], angle_bounds[1]], dtype=np.float64)
    span = high - low
    positions = low + rng.random((particles, 2)) * span
    velocities = rng.uniform(-0.25, 0.25, size=(particles, 2)) * span
    personal_best = positions.copy()
    personal_scores = np.asarray(
        [predict_design_score(pos[0], pos[1], summaries) for pos in positions],
        dtype=np.float64,
    )
    global_best = personal_best[int(np.argmax(personal_scores))].copy()
    global_score = float(np.max(personal_scores))
    trace: List[Tuple[float, float, float]] = []

    for _ in range(int(iterations)):
        r1 = rng.random((particles, 2))
        r2 = rng.random((particles, 2))
        velocities = 0.65 * velocities + 1.4 * r1 * (personal_best - positions) + 1.4 * r2 * (global_best - positions)
        velocities = np.clip(velocities, -0.25 * span, 0.25 * span)
        positions = np.clip(positions + velocities, low, high)
        scores = np.asarray(
            [predict_design_score(pos[0], pos[1], summaries) for pos in positions],
            dtype=np.float64,
        )
        improved = scores > personal_scores
        personal_best[improved] = positions[improved]
        personal_scores[improved] = scores[improved]
        best_idx = int(np.argmax(personal_scores))
        if float(personal_scores[best_idx]) > global_score:
            global_score = float(personal_scores[best_idx])
            global_best = personal_best[best_idx].copy()
        for pos, score in zip(positions, scores):
            trace.append((float(pos[0]), float(pos[1]), float(score)))

    candidates = trace + [
        (float(pos[0]), float(pos[1]), float(score))
        for pos, score in zip(personal_best, personal_scores)
    ]
    candidates.sort(key=lambda item: item[2], reverse=True)
    selected = []
    for width, angle, score in candidates:
        if any(abs(width - row["width"]) < 0.02 and abs(angle - row["angle"]) < 1.0 for row in selected):
            continue
        selected.append({"width": width, "angle": angle, "predicted_score": score})
        if len(selected) >= int(top_k):
            break
    return selected


def propose_heterogeneous_pairs(
    design_library: Sequence[Dict[str, float]],
    pair_count: int = 3,
) -> List[Dict[str, float]]:
    pairs = []
    for i, design_a in enumerate(design_library):
        for j, design_b in enumerate(design_library):
            if j <= i:
                continue
            score_a = float(design_a.get("predicted_score", 0.0))
            score_b = float(design_b.get("predicted_score", 0.0))
            diversity = abs(float(design_a["width"]) - float(design_b["width"])) + abs(
                float(design_a["angle"]) - float(design_b["angle"])
            ) / 30.0
            pair_score = 0.5 * (score_a + score_b) + 0.05 * diversity
            pairs.append(
                {
                    "layout": "heterogeneous_ab",
                    "A_width_ratio": float(design_a["width"]),
                    "A_attack_angle_deg": float(design_a["angle"]),
                    "B_width_ratio": float(design_b["width"]),
                    "B_attack_angle_deg": float(design_b["angle"]),
                    "predicted_pair_score": float(pair_score),
            "placement": "modules 1,3,5,7 = A; modules 2,4,6,8 = B",
                }
            )
    pairs.sort(key=lambda row: row["predicted_pair_score"], reverse=True)
    return pairs[: int(pair_count)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GOAL-aligned SAC training for homogeneous and heterogeneous snake scale research."
    )
    subparsers = parser.add_subparsers(dest="command")

    train = subparsers.add_parser(
        "train",
        aliases=["train-design", "evaluate-design"],
        help="Train one morphology configuration.",
    )
    train.add_argument("--layout", choices=["homogeneous", "heterogeneous_ab"], default="homogeneous")
    train.add_argument("--a-width", type=float, required=True)
    train.add_argument("--a-angle", type=float, required=True)
    train.add_argument("--b-width", type=float)
    train.add_argument("--b-angle", type=float)
    train.add_argument("--config-id", default="")
    train.add_argument("--target-z", type=float, default=-85.0)
    train.add_argument("--dmax", type=float, default=DEFAULT_DMAX_CM)
    train.add_argument("--episodes-per-terrain", type=int, default=30)
    train.add_argument("--eval-episodes-per-terrain", type=int, default=5)
    train.add_argument("--terrain-order", default="carpet,cardboard")
    train.add_argument("--episode-length", type=int, default=175)
    train.add_argument("--updates-per-episode", type=int, default=1000)
    train.add_argument("--individual-updates-per-episode", type=int)
    train.add_argument("--population-updates-per-episode", type=int)
    train.add_argument("--warmup-episodes-per-terrain", type=int, default=0)
    train.add_argument("--update-schedule", choices=["replay-ramp", "fixed"], default="replay-ramp")
    train.add_argument("--update-ramp-min", type=int, default=50)
    train.add_argument("--update-ramp-replay-divisor", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--replay-size", type=int, default=DEFAULT_REPLAY_SIZE)
    train.add_argument("--population-replay-size", type=int, default=DEFAULT_REPLAY_SIZE)
    train.add_argument("--individual-replay-size", type=int, default=DEFAULT_REPLAY_SIZE)
    train.add_argument("--profile-velocity", type=int, default=120)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--gamma", type=float, default=0.99)
    train.add_argument("--tau", type=float, default=0.01)
    train.add_argument("--alpha-init", type=float, default=0.01)
    train.add_argument("--target-entropy", type=float, default=-7.0)
    train.add_argument("--grad-clip-value", type=float, default=1.0)
    train.add_argument("--success-bonus", type=float, default=50.0)
    train.add_argument("--time-penalty", type=float, default=0.001)
    train.add_argument("--action-settle-s", type=float, default=0.3)
    train.add_argument("--seed", type=int, default=12345)
    train.add_argument("--design-id", type=int, default=0)
    train.add_argument("--use-population-pso", action="store_true")
    train.add_argument("--output-dir", default=str(Path("CoadaptationCode") / "results_goal_research"))
    train.add_argument("--resume-run-dir", type=Path)
    train.add_argument("--individual-checkpoint", type=Path)
    train.add_argument("--dry-run", action="store_true")
    train.add_argument("--hardware-disabled", action="store_true")
    train.add_argument("--interactive-reset", dest="interactive_reset", action="store_true", default=True)
    train.add_argument("--no-interactive-reset", dest="interactive_reset", action="store_false")
    train.add_argument("--terrain-change-prompt", dest="terrain_change_prompt", action="store_true", default=True)
    train.add_argument("--no-terrain-change-prompt", dest="terrain_change_prompt", action="store_false")
    train.add_argument("--print-step-rewards", dest="print_step_rewards", action="store_true", default=True)
    train.add_argument("--no-print-step-rewards", dest="print_step_rewards", action="store_false")
    train.add_argument("--eval-population-policy", dest="eval_population_policy", action="store_true", default=True)
    train.add_argument("--no-eval-population-policy", dest="eval_population_policy", action="store_false")

    propose = subparsers.add_parser("propose-library", help="Use PSO to propose homogeneous scale designs and A/B pairs.")
    propose.add_argument("--summary-csv", type=Path)
    propose.add_argument("--output-json", type=Path, default=Path("goal_scale_design_proposals.json"))
    propose.add_argument("--particles", type=int, default=24)
    propose.add_argument("--iterations", type=int, default=40)
    propose.add_argument("--top-k", type=int, default=4)
    propose.add_argument("--pair-count", type=int, default=3)
    propose.add_argument("--seed", type=int, default=12345)
    pso = subparsers.add_parser("propose-pso", help="Use a trained population critic to propose designs.")
    pso.add_argument("--layout", choices=["homogeneous", "heterogeneous_ab"], default="homogeneous")
    pso.add_argument("--a-width", type=float, default=0.675)
    pso.add_argument("--a-angle", type=float, default=15.0)
    pso.add_argument("--b-width", type=float)
    pso.add_argument("--b-angle", type=float)
    pso.add_argument("--config-id", default="population_pso")
    pso.add_argument("--target-z", type=float, default=-85.0)
    pso.add_argument("--dmax", type=float, default=DEFAULT_DMAX_CM)
    pso.add_argument("--episodes-per-terrain", type=int, default=0)
    pso.add_argument("--eval-episodes-per-terrain", type=int, default=0)
    pso.add_argument("--terrain-order", default="carpet,cardboard")
    pso.add_argument("--episode-length", type=int, default=175)
    pso.add_argument("--updates-per-episode", type=int, default=1000)
    pso.add_argument("--individual-updates-per-episode", type=int)
    pso.add_argument("--population-updates-per-episode", type=int)
    pso.add_argument("--warmup-episodes-per-terrain", type=int, default=0)
    pso.add_argument("--update-schedule", choices=["replay-ramp", "fixed"], default="replay-ramp")
    pso.add_argument("--update-ramp-min", type=int, default=50)
    pso.add_argument("--update-ramp-replay-divisor", type=int, default=4)
    pso.add_argument("--batch-size", type=int, default=32)
    pso.add_argument("--replay-size", type=int, default=DEFAULT_REPLAY_SIZE)
    pso.add_argument("--population-replay-size", type=int, default=DEFAULT_REPLAY_SIZE)
    pso.add_argument("--individual-replay-size", type=int, default=DEFAULT_REPLAY_SIZE)
    pso.add_argument("--profile-velocity", type=int, default=120)
    pso.add_argument("--learning-rate", type=float, default=1e-3)
    pso.add_argument("--gamma", type=float, default=0.99)
    pso.add_argument("--tau", type=float, default=0.01)
    pso.add_argument("--alpha-init", type=float, default=0.01)
    pso.add_argument("--target-entropy", type=float, default=-7.0)
    pso.add_argument("--grad-clip-value", type=float, default=1.0)
    pso.add_argument("--success-bonus", type=float, default=50.0)
    pso.add_argument("--time-penalty", type=float, default=0.001)
    pso.add_argument("--action-settle-s", type=float, default=0.3)
    pso.add_argument("--seed", type=int, default=12345)
    pso.add_argument("--design-id", type=int, default=0)
    pso.add_argument("--output-dir", default=str(Path("CoadaptationCode") / "results_goal_research"))
    pso.add_argument("--dry-run", action="store_true", default=True)
    pso.add_argument("--hardware-disabled", action="store_true", default=True)
    pso.add_argument("--interactive-reset", dest="interactive_reset", action="store_true", default=False)
    pso.add_argument("--use-population-pso", action="store_true", default=True)
    pso.add_argument("--pso-score-mode", choices=["robust", "terrain"], default="robust")
    pso.add_argument("--pso-terrain", choices=TERRAIN_LABELS, default="carpet")
    pso.add_argument("--homogeneous-particles", type=int, default=24)
    pso.add_argument("--heterogeneous-particles", type=int, default=24)
    pso.add_argument("--iterations", type=int, default=40)
    pso.add_argument("--top-k", type=int, default=4)
    pso.add_argument("--pair-count", type=int, default=3)
    pso.add_argument("--pso-batch-size", type=int, default=256)
    pso.add_argument("--population-checkpoint", type=Path)
    pso.add_argument("--population-replay-npz", type=Path)
    pso.add_argument("--output-json", type=Path, default=Path("goal_population_pso_proposals.json"))
    return parser


def train_main(args: argparse.Namespace) -> Dict[str, object]:
    runner = GoalResearchRunner(args)
    if args.individual_checkpoint and not getattr(args, "resume_run_dir", None):
        checkpoint = load_torch_payload(args.individual_checkpoint)
        payload = checkpoint["individual_agent"]
        runner.population_agent.load_state_dict_payload(payload, load_optimizers=False)
        runner.individual_agent.load_state_dict_payload(payload, load_optimizers=False)
        print(f"Warm-started population and individual SAC from {args.individual_checkpoint}.")
    summary = runner.run()
    print(json.dumps(summary, indent=2))
    return summary


def evaluate_design_main(args: argparse.Namespace) -> Dict[str, object]:
    if not args.individual_checkpoint and not getattr(args, "resume_run_dir", None):
        raise ValueError("evaluate-design requires --individual-checkpoint or --resume-run-dir.")
    runner = GoalResearchRunner(args)
    if args.individual_checkpoint:
        checkpoint = load_torch_payload(args.individual_checkpoint)
        payload = checkpoint["individual_agent"]
        checkpoint_terrain = str(checkpoint.get("terrain", ""))
        if checkpoint_terrain:
            runner.trained_individual_payloads[checkpoint_terrain] = payload
        else:
            for terrain in runner._terrain_order():
                runner.trained_individual_payloads[terrain] = payload
    summary = runner.evaluate()
    runner.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def propose_library_main(args: argparse.Namespace) -> Dict[str, object]:
    summaries = load_morphology_summary_csv(args.summary_csv) if args.summary_csv else []
    homogeneous_library = run_pso_design_library(
        summaries=summaries,
        particles=args.particles,
        iterations=args.iterations,
        top_k=args.top_k,
        seed=args.seed,
    )
    heterogeneous_pairs = propose_heterogeneous_pairs(homogeneous_library, pair_count=args.pair_count)
    payload = {
        "homogeneous_library": homogeneous_library,
        "heterogeneous_pairs": heterogeneous_pairs,
        "notes": "PSO uses morphology summary scores only; policy observations do not include scale features.",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def propose_pso_main(args: argparse.Namespace) -> Dict[str, object]:
    if not args.population_checkpoint:
        raise ValueError("propose-pso requires --population-checkpoint.")
    runner = GoalResearchRunner(args)
    checkpoint = load_torch_payload(args.population_checkpoint)
    runner.population_agent.load_state_dict_payload(checkpoint["population_agent"], load_optimizers=False)
    if not args.population_replay_npz:
        replay_path = str(checkpoint.get("run_data", {}).get("population_replay", ""))
        if replay_path:
            args.population_replay_npz = Path(replay_path)
    if not args.population_replay_npz:
        raise ValueError("propose-pso requires --population-replay-npz or a checkpoint with run_data.population_replay.")
    if not args.population_replay_npz.exists():
        raise ValueError(f"Population replay snapshot does not exist: {args.population_replay_npz}")
    runner.population_replay.load_npz(args.population_replay_npz)
    payload = runner.propose_pso_designs()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def main(argv: Optional[Sequence[str]] = None):
    argv_list = list(sys.argv[1:] if argv is None else argv)
    known_commands = {"train", "train-design", "evaluate-design", "propose-library", "propose-pso", "-h", "--help"}
    if not argv_list or argv_list[0] not in known_commands:
        argv_list = ["train", *argv_list]
    parser = build_parser()
    args = parser.parse_args(argv_list)
    if args.command in {"train", "train-design"}:
        return train_main(args)
    if args.command == "evaluate-design":
        return evaluate_design_main(args)
    if args.command == "propose-library":
        return propose_library_main(args)
    if args.command == "propose-pso":
        return propose_pso_main(args)
    parser.error(f"Unknown command: {args.command}")
    return None


if __name__ == "__main__":
    main()