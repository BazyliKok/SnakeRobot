import json
import gymnasium as gym
import matplotlib.pyplot as plt
from soft_actor_critic_coadapt import SoftActorCriticCoadapt
from snakeenv_thread_coadapt import MotorFaultError, SnakeEnv
import numpy as np
from replaybuffercoadapt import CoadaptReplayBuffer
import os
import torch
from scipy.interpolate import interp1d
import threading
import matplotlib.pyplot as plt
import pandas as pd
import os
import utils
import pickle
import gc
import random
from pso_batch import PSO_batch
import time
import rlkit.torch.pytorch_util as ptu
import rlkit.torch.networks as rlkit_networks
from motorssynced import MotorsSynced
from collections import Counter

from datetime import datetime

def identity(x):
    return x

# Older saved rlkit modules reference rlkit.torch.networks symbols that are
# missing from newer RLKit forks. Keep them available before loading trusted
# checkpoints.
rlkit_networks.identity = identity
if (
    not hasattr(rlkit_networks, 'FlattenMlp')
    and hasattr(rlkit_networks, 'ConcatMlp')
):
    rlkit_networks.FlattenMlp = rlkit_networks.ConcatMlp

class Train():
    LOSS_DIAGNOSTIC_KEYS = [
        'Q Target Mean',
        'Q Target Std',
        'Q1 Prediction Mean',
        'Q2 Prediction Mean',
        'Q1 TD Abs Mean',
        'Q2 TD Abs Mean',
        'Q1 TD Abs Max',
        'Q2 TD Abs Max',
        'Alpha',
    ]

    def __init__(self):        

        self.env = gym.make("SnakeRobot")
    
        self._reward_scale = 1.0
        self.optimized_params = None
        self._episode_length = 175 # number of timesteps per episode
        self.episode_counter = None
        self.policy_action_warmup_episodes = 2  # full-random episodes before policy/random mixing starts
        self.training_update_warmup_episodes = 1  # collected episodes before SAC updates start
        self.design_cylces = 20 # total number of design cycles
        self.design_mode = os.getenv('SNAKE_SCALE_DESIGN_MODE', 'homogeneous').strip().lower()
        self.initial_designs = SnakeEnv.get_init_design_parameters(self.design_mode)
        self.terrain_sequence = self._parse_active_terrains()
        self.training_terrain_block_size = max(
            1,
            int(os.getenv('SNAKE_EPISODES_PER_TERRAIN', '50')),
        )
        self.episode_iterations = len(self.terrain_sequence) * self.training_terrain_block_size # number of episodes per design
        default_results_tag = f"scale_ab_{'_'.join(self.terrain_sequence)}"
        self.results_tag = os.getenv('SNAKE_RESULTS_TAG', default_results_tag)
        self.legacy_results_tags = [self.results_tag] + self.terrain_sequence + ['carton']
        resuming_from_checkpoint = self._read_bool_env('SNAKE_RESUME_CHECKPOINT', default=False)
        self.use_legacy_policy_warm_start = self._read_bool_env(
            'SNAKE_USE_LEGACY_POLICY',
            default=False,
        )
        if self.use_legacy_policy_warm_start:
            raise RuntimeError(
                "Legacy warm-start is disabled for continuous scale-parameter training. "
                "Start fresh or resume a checkpoint created with the scale-parameter schema."
            )
        self.legacy_results_dir = os.getenv(
            'SNAKE_LEGACY_RESULTS_DIR',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
        )
        self.legacy_checkpoint_prefix = os.getenv(
            'SNAKE_LEGACY_CHECKPOINT_PREFIX',
            '2025_03_17-12_46_29_Design3_ep52',
        )
        self.terrain_name_to_id = dict(SnakeEnv.terrain_name_to_id)
        self.training_terrain_block_order = []
        self.training_episode_schedule = []
        self.training_schedule_design_counter = None
        self.randomize_terrain_order = self._read_bool_env('SNAKE_RANDOMIZE_TERRAIN_ORDER', default=False)
        self.current_training_terrain = None
        self.current_training_terrain_id = -1
        self.current_training_block_index = -1
        self.current_training_episode_in_block = -1
        self.current_schedule_seed = None
        self._last_seed_context = None
        self.current_episode_seed = None
        self.current_update_seed = None
        self.current_design_optimization_seed = None
        self._last_individual_reset_key = None

        # Keep some exploration for fresh runs. The continuous scale-parameter
        # experiment intentionally starts from scratch by default.
        default_action_noise_std = '0.02'
        default_repeat_action_eps = '0.02'
        default_repeat_action_perturb_std = '0.02'
        self.action_noise_std = max(
            0.0,
            float(os.getenv('SNAKE_ACTION_NOISE_STD', default_action_noise_std)),
        )
        self.repeat_action_eps = max(
            0.0,
            float(os.getenv('SNAKE_REPEAT_ACTION_EPS', default_repeat_action_eps)),
        )
        self.repeat_action_perturb_std = max(
            0.0,
            float(os.getenv('SNAKE_REPEAT_ACTION_PERTURB_STD', default_repeat_action_perturb_std)),
        )
        self.random_action_prob_start = 0.2
        self.random_action_prob_decay = 0.05
        self.random_action_prob_min = 0.0
        default_pop_train_start_design = '0'
        default_terrain_prefill_episodes = '0'
        default_policy_warmup_episodes = str(self.policy_action_warmup_episodes)
        default_update_warmup_episodes = str(self.training_update_warmup_episodes)
        self.population_training_start_design = int(
            os.getenv('SNAKE_POP_TRAIN_START_DESIGN', default_pop_train_start_design)
        )
        self.terrain_prefill_episodes = max(
            0,
            int(os.getenv('SNAKE_TERRAIN_PREFILL_EPISODES', default_terrain_prefill_episodes)),
        )
        self.policy_action_warmup_episodes = max(
            0,
            int(os.getenv('SNAKE_POLICY_WARMUP_EPISODES', default_policy_warmup_episodes)),
        )
        self.training_update_warmup_episodes = max(
            0,
            int(os.getenv('SNAKE_UPDATE_WARMUP_EPISODES', default_update_warmup_episodes)),
        )
        self.max_motor_fault_step_retries = max(
            0,
            int(os.getenv('SNAKE_MOTOR_FAULT_STEP_RETRIES', '3')),
        )
        self.motor_fault_step_retry_delay_s = max(
            0.0,
            float(os.getenv('SNAKE_MOTOR_FAULT_STEP_RETRY_DELAY_S', '1.0')),
        )
        self.episodeCumulativeRewards = []  # Stores cumulative rewards per episode
        self.cumulativeRewards = []  # Stores cumulative rewards per step

        self.episodeCumulativeRewards = []

        self.eachEpisodeCumuRewards = []

        self.num_init_designs = len(self.initial_designs) # number of initial design cycles
        self.seed = int(os.getenv('SNAKE_EXPERIMENT_SEED', '12345'))
        self.eval_episodes_per_terrain = max(0, int(os.getenv('SNAKE_EVAL_EPISODES_PER_TERRAIN', '3')))
        self.training_score_last_episodes_per_terrain = max(
            1,
            int(os.getenv('SNAKE_TRAINING_SCORE_LAST_EPISODES_PER_TERRAIN', '1')),
        )
        self.eval_robustness_lambda = 0.5
        # set up replay
        self.replay = CoadaptReplayBuffer(
            max_replay_buffer_size_species=int(1e6),
            max_replay_buffer_size_population=int(1e7),
            env= self.env,
            env_info_sizes=None
        )
        self._refresh_active_terrain_filter()
        self._seed_global_rngs('initialization')

        # set up RL algorithm
        self.rl_method = SoftActorCriticCoadapt
        self.networks = self.rl_method.create_networks(env=self.env)
        self.rl_alg = self.rl_method(env=self.env, replay=self.replay, networks=self.networks)
        if self.use_legacy_policy_warm_start:
            self.warm_start_from_legacy_checkpoint()

        # set up design variables
        self.do_alg = PSO_batch(self.replay, self.env)
        self.design_counter = 0
        self.data_design_type = 'Initial'
        

        self.date = datetime.now().strftime("%Y_%m_%d") # for files
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def _read_bool_env(self, name, default):
        env_value = os.getenv(name)
        if env_value is None:
            return default
        return env_value.strip().lower() in ('1', 'true', 'yes', 'on')

    def _parse_active_terrains(self, raw_value=None):
        raw_value = os.getenv('SNAKE_ACTIVE_TERRAINS', 'carpet,foam') if raw_value is None else raw_value
        terrain_sequence = [terrain.strip() for terrain in str(raw_value).split(',') if terrain.strip()]
        if not terrain_sequence:
            raise ValueError(
                "SNAKE_ACTIVE_TERRAINS must name at least one terrain, "
                "for example 'carpet,foam'."
            )

        invalid_terrains = [terrain for terrain in terrain_sequence if terrain not in SnakeEnv.terrains]
        if invalid_terrains:
            raise ValueError(
                f"Unknown terrain(s) in SNAKE_ACTIVE_TERRAINS: {invalid_terrains}. "
                f"Use one or more of {SnakeEnv.terrains}."
            )
        return terrain_sequence

    def _active_terrain_ids(self):
        return [self.terrain_name_to_id[terrain] for terrain in self.terrain_sequence]

    def _refresh_active_terrain_filter(self):
        if hasattr(self, 'replay') and self.replay is not None:
            self.replay.set_active_terrain_ids(self._active_terrain_ids())

    def _scale_design_summary_fields(self, design=None):
        design = SnakeEnv._coerce_design_vector(SnakeEnv.get_current_design() if design is None else design)
        summary = SnakeEnv.design_summary(design)
        summary['Scale_Design_Mode'] = self.design_mode
        summary['Design_Config'] = '|'.join(f'{value:.6g}' for value in design)
        return summary

    def _action_dim(self):
        action_shape = getattr(self.env.action_space, "shape", None)
        if action_shape and len(action_shape) > 0:
            return action_shape[0]
        return 7

    def _random_action_probability(self):
        episode_idx = 0 if self.current_training_episode_in_block is None else int(self.current_training_episode_in_block)
        return max(
            self.random_action_prob_min,
            self.random_action_prob_start - (self.random_action_prob_decay * episode_idx),
        )

    def _should_use_policy_actions(self):
        episode_idx = 0 if self.episode_counter is None else int(self.episode_counter)
        return episode_idx >= self.policy_action_warmup_episodes

    def _should_train_updates(self):
        episode_idx = 0 if self.episode_counter is None else int(self.episode_counter)
        if episode_idx < self.training_update_warmup_episodes:
            return False

        episode_in_block = (
            0 if self.current_training_episode_in_block is None
            else int(self.current_training_episode_in_block)
        )
        return episode_in_block >= self.terrain_prefill_episodes

    def _is_terrain_prefill_episode(self):
        episode_in_block = (
            0 if self.current_training_episode_in_block is None
            else int(self.current_training_episode_in_block)
        )
        return episode_in_block < self.terrain_prefill_episodes

    def _stable_seed(self, *components):
        modulus = 2 ** 32
        seed = int(self.seed) % modulus
        for component in components:
            token = f'|{component!r}|'
            for char in token:
                seed = ((seed * 131) + ord(char)) % modulus
        return int(seed)

    def _seed_global_rngs(self, *components):
        seed = self._stable_seed(*components)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._last_seed_context = {
            'seed': int(seed),
            'components': [str(component) for component in components],
        }
        return int(seed)

    def _initialize_run_logs(self):
        self.stateList = []
        self.actionList = [[] for _ in range(self._action_dim())]
        self.designList = [[] for _ in range(len(SnakeEnv.design_parameter_bounds))]
        self.timestepRewards = []
        self.episodeCumulativeRewards = []
        self.cumulativeRewards = []
        self.epList = []
        self.timesteps = []
        self.epListLoss = []
        self.q1loss = []
        self.q2loss = []
        self.policyloss = []
        self.popq1loss = []
        self.popq2loss = []
        self.poppolicyloss = []
        self.ind_loss_diagnostics = {key: [] for key in self.LOSS_DIAGNOSTIC_KEYS}
        self.pop_loss_diagnostics = {key: [] for key in self.LOSS_DIAGNOSTIC_KEYS}
        self.progressRewardComponents = []
        self.distanceProgressCmComponents = []
        self.rawDistanceProgressCmComponents = []
        self.windowProgressCmComponents = []
        self.xDriftPenaltyComponents = []
        self.headingPenaltyComponents = []
        self.livingPenaltyComponents = []
        self.noProgressPenaltyComponents = []
        self.backwardPenaltyComponents = []

    def _upsert_csv_rows(self, path, frame, key_columns):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        frame = frame.copy()

        if not os.path.isfile(path):
            frame.to_csv(path, index=False)
            return

        try:
            existing = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            frame.to_csv(path, index=False)
            return

        for column in frame.columns:
            if column not in existing.columns:
                existing[column] = np.nan
        for column in existing.columns:
            if column not in frame.columns:
                frame[column] = np.nan

        ordered_columns = list(existing.columns)
        existing_keys = existing[key_columns].astype(str).agg('||'.join, axis=1)
        incoming_keys = set(frame[key_columns].astype(str).agg('||'.join, axis=1))
        if incoming_keys:
            existing = existing[~existing_keys.isin(incoming_keys)]

        updated = pd.concat([existing[ordered_columns], frame[ordered_columns]], ignore_index=True)
        updated.to_csv(path, index=False)

    def _set_output_filenames(self):
        self.date = datetime.now().strftime("%Y_%m_%d")
        name = "Rewards_DesignCycle{}_{}".format(str(self.design_counter), self.results_tag)
        self.filename = self.date+name
        name = "Losses_DesignCycle{}_{}".format(str(self.design_counter), self.results_tag)
        self.lossFilename = self.date+name

    def _checkpoint_results_dir(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_bazyli')

    def _print_design_installation_prompt(self, design):
        design = SnakeEnv._coerce_design_vector(design)
        summary = SnakeEnv.design_summary(design)
        print('')
        print('=== INSTALL SCALE CONFIGURATION ===')
        print(f'Design cycle {self.design_counter}')
        print(f'Mode: {self.design_mode}')
        print(
            f"Scale A: width_ratio={summary['A_Width_Ratio']:.3f}, "
            f"actual_width={summary['A_Actual_Width']:.3f}, "
            f"attack_angle_deg={summary['A_Attack_Angle_Deg']:.2f}"
        )
        print(
            f"Scale B: width_ratio={summary['B_Width_Ratio']:.3f}, "
            f"actual_width={summary['B_Actual_Width']:.3f}, "
            f"attack_angle_deg={summary['B_Attack_Angle_Deg']:.2f}"
        )
        print('Modules 1,3,5,7: Scale A')
        print('Modules 2,4,6,8: Scale B')
        print(f"Active terrains: {', '.join(self.terrain_sequence)}")
        print(f"Episodes per terrain: {self.training_terrain_block_size}")
        print('===================================')
        print('')

    def _format_design_for_terminal(self, design):
        return SnakeEnv.format_design_for_terminal(design)

    def _active_replay_indices(self, buffer):
        active_size = int(buffer._size)
        max_size = int(buffer._max_replay_buffer_size)
        if active_size <= 0:
            return np.array([], dtype=np.int64)
        if active_size < max_size and buffer._top == active_size:
            return np.arange(active_size, dtype=np.int64)
        start = int(buffer._top)
        return np.concatenate([
            np.arange(start, max_size, dtype=np.int64),
            np.arange(0, start, dtype=np.int64),
        ])[:active_size]

    def _training_rollouts_from_population_replay(self):
        buffer = self.replay._population_buffer
        active_indices = self._active_replay_indices(buffer)
        if len(active_indices) == 0:
            return []

        terrain_info = None
        if 'terrain_id' in buffer._env_info_keys:
            terrain_info = buffer._env_infos['terrain_id'].reshape(-1)

        rollouts = []
        current_return = 0.0
        current_length = 0
        current_terrain_id = None

        def _finish_rollout():
            nonlocal current_return, current_length, current_terrain_id
            if current_length <= 0:
                return
            terrain_id = -1 if current_terrain_id is None else int(current_terrain_id)
            rollouts.append({
                'terrain_id': terrain_id,
                'terrain': SnakeEnv.terrain_id_to_name.get(terrain_id, 'unknown'),
                'return': float(current_return),
                'length': int(current_length),
            })
            current_return = 0.0
            current_length = 0
            current_terrain_id = None

        for idx in active_indices:
            terrain_id = -1
            if terrain_info is not None:
                terrain_id = int(terrain_info[idx])

            if current_length > 0 and current_terrain_id != terrain_id:
                _finish_rollout()

            if current_terrain_id is None:
                current_terrain_id = terrain_id

            current_return += float(buffer._rewards[idx].reshape(-1)[0])
            current_length += 1

            if bool(buffer._terminals[idx].reshape(-1)[0]):
                _finish_rollout()

        _finish_rollout()
        return rollouts[-self.episode_iterations:]

    def _summarize_training_performance(self):
        rollouts = self._training_rollouts_from_population_replay()
        current_design = SnakeEnv.get_current_design()
        selected_returns_by_terrain = {}
        selected_lengths_by_terrain = {}

        for terrain in self.terrain_sequence:
            terrain_rollouts = [
                rollout for rollout in rollouts
                if rollout['terrain'] == terrain
            ]
            selected = terrain_rollouts[-self.training_score_last_episodes_per_terrain:]
            selected_returns_by_terrain[terrain] = [rollout['return'] for rollout in selected]
            selected_lengths_by_terrain[terrain] = [rollout['length'] for rollout in selected]

        def _finite_mean(values):
            arr = np.asarray(values, dtype=np.float32)
            arr = arr[np.isfinite(arr)]
            return float(np.mean(arr)) if len(arr) > 0 else np.nan

        terrain_means = {
            terrain: _finite_mean(selected_returns_by_terrain[terrain])
            for terrain in self.terrain_sequence
        }
        terrain_lengths = {
            terrain: _finite_mean(selected_lengths_by_terrain[terrain])
            for terrain in self.terrain_sequence
        }
        valid_terrain_returns = np.asarray(
            [value for value in terrain_means.values() if np.isfinite(value)],
            dtype=np.float32,
        )
        if len(valid_terrain_returns) > 0:
            mean_return = float(np.mean(valid_terrain_returns))
            worst_terrain_return = float(np.min(valid_terrain_returns))
            std_across_terrains = float(np.std(valid_terrain_returns))
        else:
            mean_return = np.nan
            worst_terrain_return = np.nan
            std_across_terrains = np.nan
        robustness_score = float(mean_return - self.eval_robustness_lambda * std_across_terrains)

        summary_row = {
            'Date': self.date,
            'Experiment_Seed': int(self.seed),
            'Training_Schedule_Seed': int(self.current_schedule_seed) if self.current_schedule_seed is not None else None,
            'Design_Counter': int(self.design_counter),
            'Episode_Counter': int(self.episode_counter),
            'Training_Episodes_Per_Design': int(self.episode_iterations),
            'Training_Terrain_Block_Size': int(self.training_terrain_block_size),
            'Training_Terrain_Block_Order': '|'.join(self.training_terrain_block_order),
            'Eval_Episodes_Per_Terrain': 0,
            'Score_Source': 'training_replay',
            'Training_Score_Last_Episodes_Per_Terrain': int(self.training_score_last_episodes_per_terrain),
            'Overall_Total_Eval_Rollouts': 0,
            'Overall_Valid_Eval_Rollouts': 0,
            'Overall_Reset_Failures': 0,
            'Overall_Motor_Faults': 0,
            'Overall_Failed_Eval_Rollouts': 0,
            'Mean_Return': mean_return,
            'Std_Across_Terrains': std_across_terrains,
            'Worst_Terrain_Return': worst_terrain_return,
            'Overall_Mean_Return': mean_return,
            'Std_Across_Terrain_Means': std_across_terrains,
            'Robustness_Lambda': float(self.eval_robustness_lambda),
            'Robustness_Score': robustness_score,
        }
        summary_row.update(self._scale_design_summary_fields(current_design))

        for terrain in self.terrain_sequence:
            returns = selected_returns_by_terrain[terrain]
            lengths = selected_lengths_by_terrain[terrain]
            summary_row[f'{terrain}_Training_Rollouts_Used'] = int(len(returns))
            summary_row[f'{terrain}_Mean_Return'] = terrain_means[terrain]
            summary_row[f'{terrain}_Mean_Length'] = terrain_lengths[terrain]
            summary_row[f'{terrain}_Last_Return'] = float(returns[-1]) if returns else np.nan

        results_dir = self._checkpoint_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        summary_csv_path = os.path.join(results_dir, f'{self.date}_design_eval_summary.csv')
        self._upsert_csv_rows(
            summary_csv_path,
            pd.DataFrame([summary_row]),
            ['Date', 'Design_Counter', 'Episode_Counter', 'Score_Source'],
        )
        detail_json_path = os.path.join(
            results_dir,
            f'{self.date}_DesignCycle{self.design_counter}_ep{self.episode_counter}_training_score_summary.json'
        )
        def _json_safe(obj):
            if isinstance(obj, dict):
                return {key: _json_safe(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [_json_safe(value) for value in obj]
            if isinstance(obj, tuple):
                return [_json_safe(value) for value in obj]
            if isinstance(obj, np.ndarray):
                return [_json_safe(value) for value in obj.tolist()]
            if isinstance(obj, np.generic):
                obj = obj.item()
            if isinstance(obj, float) and not np.isfinite(obj):
                return None
            return obj

        with open(detail_json_path, 'w') as f:
            json.dump(_json_safe({
                'summary': summary_row,
                'training_rollouts_used': selected_returns_by_terrain,
                'all_reconstructed_training_rollouts': rollouts,
            }), f, indent=2, allow_nan=False)

        print('')
        print('=== TRAINING PERFORMANCE SUMMARY ===')
        print(
            'Used the latest training episode(s) per terrain instead of '
            'running extra evaluation rollouts.'
        )
        print(f'Last episodes per terrain: {self.training_score_last_episodes_per_terrain}')
        print(f'Mean return across terrains: {mean_return}')
        print(f'Worst terrain return: {worst_terrain_return}')
        print(f'Robustness score: {robustness_score}')
        print('====================================')
        print('')


    def _build_randomized_training_schedule(self):
        self.current_schedule_seed = self._stable_seed('terrain_schedule', self.design_counter)
        if self.randomize_terrain_order:
            schedule_rng = np.random.default_rng(self.current_schedule_seed)
            terrain_block_order = list(schedule_rng.permutation(self.terrain_sequence))
        else:
            terrain_block_order = list(self.terrain_sequence)
        episode_schedule = []
        for terrain in terrain_block_order:
            episode_schedule.extend([terrain] * self.training_terrain_block_size)
        return terrain_block_order, episode_schedule

    def _has_valid_training_schedule(self):
        expected_blocks = len(self.terrain_sequence)
        expected_episodes = expected_blocks * self.training_terrain_block_size
        if len(self.training_terrain_block_order) != expected_blocks:
            return False
        if len(self.training_episode_schedule) != expected_episodes:
            return False
        if self.randomize_terrain_order:
            if Counter(self.training_terrain_block_order) != Counter(self.terrain_sequence):
                return False
        elif list(self.training_terrain_block_order) != list(self.terrain_sequence):
            return False

        for block_idx, terrain in enumerate(self.training_terrain_block_order):
            start = block_idx * self.training_terrain_block_size
            end = start + self.training_terrain_block_size
            if self.training_episode_schedule[start:end] != [terrain] * self.training_terrain_block_size:
                return False

        return True

    def _ensure_design_training_schedule(self):
        if (
            self.training_schedule_design_counter == self.design_counter
            and self._has_valid_training_schedule()
        ):
            return

        if self.episode_counter not in (None, 0):
            print(
                'No saved terrain schedule found for this design; '
                'generating a new terrain block order for the remaining episodes.'
            )

        self.training_terrain_block_order, self.training_episode_schedule = self._build_randomized_training_schedule()
        self.training_schedule_design_counter = self.design_counter
        print(
            f"Design cycle {self.design_counter} terrain block order: "
            f"{' -> '.join(self.training_terrain_block_order)}"
        )

    def _get_training_terrain_for_episode(self, episode_idx):
        self._ensure_design_training_schedule()
        if episode_idx < 0 or episode_idx >= len(self.training_episode_schedule):
            raise IndexError(
                f'Episode index {episode_idx} is out of range for '
                f'{len(self.training_episode_schedule)} scheduled training episodes.'
            )

        terrain = self.training_episode_schedule[episode_idx]
        terrain_id = self.terrain_name_to_id[terrain]
        block_idx = episode_idx // self.training_terrain_block_size
        episode_in_block = episode_idx % self.training_terrain_block_size
        return terrain, terrain_id, block_idx, episode_in_block

    def _resolve_tagged_path(self, base_path, stem, extension):
        tagged_candidates = [
            os.path.join(base_path, f'{stem}_{tag}.{extension}')
            for tag in self.legacy_results_tags
        ]
        untagged_candidate = os.path.join(base_path, f'{stem}.{extension}')

        for candidate in tagged_candidates + [untagged_candidate]:
            if os.path.exists(candidate):
                return candidate

        if os.path.isdir(base_path):
            prefix = f'{stem}_'
            suffix = f'.{extension}'
            for filename in os.listdir(base_path):
                if filename.startswith(prefix) and filename.endswith(suffix):
                    return os.path.join(base_path, filename)

        return tagged_candidates[0]

    def _load_trusted_checkpoint(self, path, **kwargs):
        try:
            return torch.load(path, weights_only=False, **kwargs)
        except TypeError as exc:
            if 'weights_only' not in str(exc):
                raise
            return torch.load(path, **kwargs)

    def warm_start_from_legacy_checkpoint(self):
        raise RuntimeError(
            "Legacy warm-start is disabled for continuous scale-parameter training. "
            "Start fresh or resume a checkpoint created with the scale-parameter schema."
        )

    def _recover_motor_fault(self, phase, exc, disable_after_recovery=True):
        print(f"Motor fault during {phase}: {exc}")

        recovered = False
        try:
            recovered = SnakeEnv.recoverMotorFault(
                context=f"{phase}: {exc}",
                force_reboot=True,
            )
        except Exception as recovery_exc:
            print(f"Motor recovery handler raised an exception: {recovery_exc}")

        if not disable_after_recovery:
            return recovered

        torque_disabled = self._disable_motor_torque_with_recovery(
            f"{phase} recovery cleanup"
        )
        return recovered and torque_disabled

    def _recover_motor_fault_for_step_retry(self, phase, exc):
        recovered = self._recover_motor_fault(
            phase,
            exc,
            disable_after_recovery=False,
        )
        if not recovered:
            self._disable_motor_torque_with_recovery(f"{phase} failed recovery cleanup")
            return False

        try:
            torque_enabled = SnakeEnv.enableMotorTorque()
        except Exception as enable_exc:
            print(f"Failed to enable motor torque after step recovery: {enable_exc}")
            torque_enabled = False

        if not torque_enabled:
            print("Motor torque could not be enabled after recovery; forcing cleanup.")
            self._disable_motor_torque_with_recovery(f"{phase} torque-enable failure cleanup")
            return False

        if self.motor_fault_step_retry_delay_s > 0:
            time.sleep(self.motor_fault_step_retry_delay_s)
        return True

    def _step_env_with_motor_recovery(self, action, step_number):
        max_attempts = self.max_motor_fault_step_retries + 1

        for attempt_idx in range(max_attempts):
            try:
                return self.env.step(action)
            except MotorFaultError as exc:
                phase = (
                    f"training episode {self.episode_counter} step {step_number} "
                    f"attempt {attempt_idx + 1}/{max_attempts}"
                )
                if attempt_idx >= max_attempts - 1:
                    print(
                        f"Motor fault persisted at step {step_number} after "
                        f"{self.max_motor_fault_step_retries} same-step retries."
                    )
                    self._recover_motor_fault(phase, exc)
                    return None

                recovered = self._recover_motor_fault_for_step_retry(phase, exc)
                if not recovered:
                    print(
                        f"Motor recovery failed at step {step_number}; "
                        "stopping without advancing the checkpoint so this step can be retried later."
                    )
                    return None

                print(
                    f"Motor recovered; retrying training episode {self.episode_counter} "
                    f"step {step_number} with the same action "
                    f"({attempt_idx + 2}/{max_attempts})."
                )

        return None

    def _disable_motor_torque_with_recovery(self, phase):
        try:
            torque_disabled = SnakeEnv.disableMotorTorque()
            if torque_disabled:
                return True
            print(
                f"Failed to disable motor torque during {phase}; "
                "forcing DYNAMIXEL reboot."
            )
        except Exception as disable_exc:
            print(
                f"Failed to disable motor torque during {phase}: {disable_exc}. "
                "Forcing DYNAMIXEL reboot."
            )

        try:
            recovered = SnakeEnv.recoverMotorFault(
                context=f"{phase}: disable torque failed",
                force_reboot=True,
            )
        except Exception as recovery_exc:
            print(f"Motor recovery after torque-disable failure raised an exception: {recovery_exc}")
            return False

        if not recovered:
            return False

        try:
            torque_disabled = SnakeEnv.disableMotorTorque()
            if not torque_disabled:
                print(f"Motor torque still could not be disabled after reboot during {phase}.")
            return torque_disabled
        except Exception as disable_exc:
            print(f"Motor torque disable after reboot raised an exception during {phase}: {disable_exc}")
            return False

    def _reset_env_with_motor_recovery(self, seed, phase, max_attempts=2, options=None):
        last_exc = None

        for attempt_idx in range(max_attempts):
            try:
                return self.env.reset(seed=seed, options=options)
            except MotorFaultError as exc:
                last_exc = exc
                recovered = self._recover_motor_fault(
                    f"{phase} reset attempt {attempt_idx + 1}/{max_attempts}",
                    exc,
                )
                if not recovered:
                    print("Motor recovery did not report success during reset retry.")
                    break
                time.sleep(1.0)

        print(
            f"Skipping {phase} after {max_attempts} failed motor recovery attempts: "
            f"{last_exc}"
        )
        return None, {'motor_fault': 1, 'motor_fault_message': str(last_exc)}

    def run(self, stopEvent, max_design_cycles_per_run=None):
        """ Runs Fast Evolution through Actor-Critic RL algorithm.
        Process design cycles until training finishes, the user stops the
        process, or an optional per-launch design limit is reached.
        """
        self._initialize_run_logs()
        ptu.set_gpu_mode(False)

        completed_cycles = 0
        while (
            self.design_counter < self.design_cylces
            and (
                max_design_cycles_per_run is None
                or completed_cycles < max_design_cycles_per_run
            )
        ):
            self._set_output_filenames()

            try:
                if self.design_counter >= self.num_init_designs and self.optimized_params is None:
                    self.first_train_op()
                    continue

                if self.design_counter < self.num_init_designs:
                    design_cycle_completed = self.initial_design_loop()
                    if not design_cycle_completed:
                        print(
                            f"Stopping run at design cycle {self.design_counter}, "
                            f"episode {self.episode_counter}; resume will retry this episode."
                        )
                        break
                    print(f'design counter at {self.design_counter}')
                    if self.design_counter == self.num_init_designs and self.optimized_params is None:
                        self.first_train_op()
                else:
                    design_cycle_completed = self.train_loop()
                    if not design_cycle_completed:
                        print(
                            f"Stopping run at design cycle {self.design_counter}, "
                            f"episode {self.episode_counter}; resume will retry this episode."
                        )
                        break
            except MotorFaultError as exc:
                self._recover_motor_fault(
                    f"design cycle {self.design_counter}",
                    exc,
                )
                print(
                    f"Design cycle {self.design_counter} failed with an exception and "
                    "will be skipped so training can continue."
                )
                self.design_counter += 1
                self.episode_counter = 0
            except Exception:
                print(
                    f"Unexpected error in design cycle {self.design_counter}; "
                    "stopping without advancing the design counter."
                )
                raise

            completed_cycles += 1

        stopEvent.set()
        return


    

    def collect_training_experience(self):
            """ Collect training data.

            This function executes a single episode in the environment using the
            exploration strategy/mechanism and the policy.
            The data, i.e. state-action-reward-nextState, is stored in the replay
            buffer.

            """

            self.stateList = []
            self.actionList = [[] for _ in range(self._action_dim())]
            self.timestepRewards = []
            self.cumulativeRewards = []
            self.epList = []
            self.timesteps = []
            self.progressRewardComponents = []
            self.distanceProgressCmComponents = []
            self.rawDistanceProgressCmComponents = []
            self.windowProgressCmComponents = []
            self.xDriftPenaltyComponents = []
            self.headingPenaltyComponents = []
            self.livingPenaltyComponents = []
            self.noProgressPenaltyComponents = []
            self.backwardPenaltyComponents = []

            # reset environment
            terrain, terrain_idx, block_idx, episode_in_block = self._get_training_terrain_for_episode(self.episode_counter)
            self.current_training_terrain = terrain
            self.current_training_terrain_id = terrain_idx
            self.current_training_block_index = block_idx
            self.current_training_episode_in_block = episode_in_block
            self.current_episode_seed = self._seed_global_rngs(
                'train_episode',
                self.design_counter,
                self.episode_counter,
            )
            SnakeEnv.set_current_terrain(terrain)
            print(
                f"CURRENT TERRAIN: {terrain} "
                f"(block {block_idx + 1}/{len(self.terrain_sequence)}, "
                f"episode {episode_in_block + 1}/{self.training_terrain_block_size}). "
                f"Place robot on this terrain before continuing reset."
            )
            state, info = self._reset_env_with_motor_recovery(
                seed=self.current_episode_seed,
                phase=f"training episode {self.episode_counter}",
            )
            if state is None:
                print(
                    f"Skipping training episode {self.episode_counter} because "
                    "the robot could not be reset after recovery attempts."
                )
                self.episodeCumulativeRewards.append(0.0)
                self.eachEpisodeCumuRewards.append([])
                self.replay.terminate_episode()
                return False
            state_dim = len(state)
            self.stateList = [[] for _ in range(state_dim)]
            steps = 0
            episodeRewards = 0
            episodeContRewards = []
            Done = False
            prev_action = None
            random_action_prob = self._random_action_probability()
            print(f'episode random action probability: {random_action_prob:.3f}')
        
            # get policies
            self.policy = self.rl_alg.get_policy_network(self.networks['individual']) #get policy here
            self.pop_policy = self.rl_alg.get_policy_network(self.networks['population']) #get policy here

            currDesign = SnakeEnv.get_current_design()
            
            while not Done and steps < self._episode_length:
                start = time.time()
                
                self.timesteps.append(steps)


                step_number = steps + 1
                print(f'Step: {step_number}')
                #state = torch.tensor(state)
                #state = state.to(torch.float32)

                
                # exploration vs exploitation
                # keep stochasticity/noise so actions do not collapse to one extreme command
                use_policy_action = (
                    self._should_use_policy_actions()
                    and np.random.rand() > random_action_prob
                )
                if use_policy_action:
                    action, _ = self.policy.get_action(state)
                else:
                    action = np.random.uniform(-1, 1, size=self._action_dim())

                action = np.asarray(action, dtype=np.float32)
                action += np.random.normal(0.0, self.action_noise_std, size=action.shape)

                if prev_action is not None and np.max(np.abs(action - prev_action)) < self.repeat_action_eps:
                    action += np.random.normal(0.0, self.repeat_action_perturb_std, size=action.shape)

                action = np.clip(action, -1.0, 1.0)
                prev_action = action.copy()

                num_logged_actions = min(len(self.actionList), len(action))
                for i in range(num_logged_actions):
                    self.actionList[i].append(action[i])
                
        
                step_result = self._step_env_with_motor_recovery(action, step_number)
                if step_result is None:
                    try:
                        self.replay.terminate_episode()
                    except Exception as replay_exc:
                        print(f"Failed to terminate replay episode after motor fault: {replay_exc}")
                    return False
                next_state, reward, terminated, truncated, info = step_result

                
                episodeRewards += reward # accumulate rewards here to track for comparison
        
                # log rewards
                self.timestepRewards.append(reward)
                self.cumulativeRewards.append(episodeRewards)
                self.epList.append(self.currEp) # to make note of what episode we are on
                self.progressRewardComponents.append(float(info.get('progress_reward', np.nan)))
                self.distanceProgressCmComponents.append(float(info.get('distance_progress_cm', np.nan)))
                self.rawDistanceProgressCmComponents.append(float(info.get('raw_distance_progress_cm', np.nan)))
                self.windowProgressCmComponents.append(float(info.get('window_progress_cm', np.nan)))
                self.xDriftPenaltyComponents.append(float(info.get('x_drift_penalty', np.nan)))
                self.headingPenaltyComponents.append(float(info.get('heading_penalty', np.nan)))
                self.livingPenaltyComponents.append(float(info.get('living_penalty', np.nan)))
                self.noProgressPenaltyComponents.append(float(info.get('no_progress_penalty', info.get('stagnation_penalty', np.nan))))
                self.backwardPenaltyComponents.append(float(info.get('backward_penalty', np.nan)))
                for i in range(len(state)):
                    self.stateList[i].append(state[i])


             

                steps += 1
                Done = terminated or truncated or (steps >= self._episode_length)
                terminal = np.array([Done]) # turn into array for replay buffer
                reward = np.array([reward])
                
                # add replay sample

                print(f'action shape: {action.shape}')
                self.replay.add_sample(observation=state, action=action, reward=reward, next_observation=next_state,
                   terminal=terminal, env_info={'terrain_id': terrain_idx})

                state = next_state # set state for next iteration

            if self._disable_motor_torque_with_recovery("end of training episode"):
                print('disabled torque')
            else:
                print('Motor torque disable/recovery failed at end of training episode.')


                
             
            self.episodeCumulativeRewards.append(episodeRewards)
            self.eachEpisodeCumuRewards.append(episodeContRewards) # list of a list

            self.logData() # log data
            self.replay.terminate_episode() # run replay end sequence
            return True




    def initialize_episode(self):
        """Initialize training state before entering a design's episode loop.

        Individual policy/replay resets are handled at terrain-block
        boundaries so terrain adaptation does not leak into the next block.

        """
        if not self._reset_individual_for_terrain_block_if_needed():
            self.rl_alg.episode_init(copy_population_to_individual=False)


        self.data_rewards = []

    def _reset_individual_for_terrain_block_if_needed(self):
        episode_idx = 0 if self.episode_counter is None else int(self.episode_counter)
        block_size = max(1, int(self.training_terrain_block_size))
        if episode_idx % block_size != 0:
            return False

        block_idx = episode_idx // block_size
        reset_key = (int(self.design_counter), int(block_idx))
        if reset_key == self._last_individual_reset_key:
            return False

        print(
            f"Resetting individual policy from population for design cycle "
            f"{self.design_counter}, terrain block {block_idx + 1}."
        )
        self.rl_alg.episode_init(copy_population_to_individual=True)
        self.replay.reset_individual_buffer()
        self._last_individual_reset_key = reset_key
        return True
    
    def first_train_op(self):
        print('in first train op')
        self.data_design_type = 'Optimized'

        print(f'design counter at {self.design_counter}')
        if self.design_counter >= self.num_init_designs: # first optimized design after init loop or resume
         
            self.current_design_optimization_seed = self._seed_global_rngs(
                'design_opt_bootstrap',
                self.design_counter,
                self.episode_counter,
            )
            self.optimized_params = SnakeEnv.get_default_design()

            q_network = self.rl_alg.get_q_network(self.networks['population'])
            policy_network = self.rl_alg.get_policy_network(self.networks['population'])
            self.cost, self.optimized_params = self.do_alg.optimize_design(
                design=self.optimized_params,
                q_network=q_network,
                policy_network=policy_network,
                active_terrains=self.terrain_sequence,
                design_mode=self.design_mode,
            )
            self.optimized_params = SnakeEnv._coerce_design_vector(self.optimized_params)
            print('OPTIMIZED PARAM NEW DESIGN: ', self.optimized_params)
            print('COST: ', self.cost)
            self.save_networks()
        



    def train_loop(self):
        """ Runs the Fast Evolution through Actor-Critic RL algorithm.

        First the initial design loop is executed in which the rl-algorithm
        is exeuted on the initial designs. Then the design-optimization
        process starts.
        It is possible to have different numbers of iterations for initial
        designs and the design optimization process.
        """
       
        iterations = self.episode_iterations 
        self.data_design_type = 'Optimized'
        if self.optimized_params is None:
            raise ValueError(
                "No optimized design is available for train_loop. "
                "Run first_train_op or resume from a checkpoint that contains optimized_params."
            )
        SnakeEnv.set_new_design(self.optimized_params)
        self._print_design_installation_prompt(self.optimized_params)
        self.initialize_episode()
        self._ensure_design_training_schedule()

        # Reinforcement Learning
        start_ep = self.episode_counter
        for episode in range(start_ep, iterations):
            print('IN TRAINING LOOP')
            self.currEp = episode
            if not self.train_single_iteration():
                print("Stopping training loop because no experience was collected.")
                return False
        
            #self.plot_rewards()

        # Score current design before running design optimization.
        # By default this uses the training replay and runs no extra rollouts.
        self.evaluate_policy()

        # Design Optimization
        print(f'design counter at {self.design_counter}')
        if self.design_counter >= self.num_init_designs:
            self._data_design_type = 'Optimized'
            self.current_design_optimization_seed = self._seed_global_rngs(
                'design_opt',
                self.design_counter,
                self.episode_counter,
            )
            q_network = self.rl_alg.get_q_network(self.networks['population'])
            policy_network = self.rl_alg.get_policy_network(self.networks['population'])
            self.cost, self.optimized_params = self.do_alg.optimize_design(
                design=self.optimized_params,
                q_network=q_network,
                policy_network=policy_network,
                active_terrains=self.terrain_sequence,
                design_mode=self.design_mode,
            )
            self.optimized_params = SnakeEnv._coerce_design_vector(self.optimized_params)
            print('NEW DESIGN PARAMETERS: ',self.optimized_params)
            print('COST: ', self.cost)
        #else: # randomize next design
        #    self._data_design_type = 'Random'
        #    self.optimized_params = SnakeEnv.get_random_design()
        #    self.optimized_params = list(self.optimized_params)

        
        self.design_counter += 1 # another design
        self.episode_counter = 0
        self.save_networks()

        return True
            
            
    def train_single_iteration(self):
        experience_collected = False

        self._reset_individual_for_terrain_block_if_needed()
        self.replay.set_mode("species")
        try:
            experience_collected = self.collect_training_experience() # collect data
        except Exception as exc:
            self._recover_motor_fault(
                f"training episode {self.episode_counter} collection",
                exc,
            )
            try:
                self.replay.terminate_episode()
            except Exception as replay_exc:
                print(f"Failed to terminate replay episode after exception: {replay_exc}")
            print("Skipping the rest of this training episode after an unexpected exception.")
            experience_collected = False

        if not experience_collected:
            print(
                f"No experience collected for episode {self.episode_counter}; "
                "leaving episode counter unchanged and not saving a new checkpoint."
            )
            return False
        
        train_pop = self.design_counter >= self.population_training_start_design
        print(
            f'population SAC training enabled: {train_pop} '
            f'(start design {self.population_training_start_design})'
        )
        
        print('train single iteration check if training warmup is complete')
        if experience_collected and self._should_train_updates():  # can start training after enough full episodes are collected
            print('training warmup complete')
            self.current_update_seed = self._seed_global_rngs(
                'train_update',
                self.design_counter,
                self.episode_counter,
            )
            q1loss, q2loss, policyloss, popq1loss, popq2loss, poppolicyloss = self.rl_alg.single_train_step(train_ind=True, train_pop=train_pop) # train one step
            
            #log data on lists
            self.q1loss.extend(q1loss)
            self.q2loss.extend(q2loss)
            self.policyloss.extend(policyloss) 
            self.popq1loss.extend(popq1loss)
            self.popq2loss.extend(popq2loss)
            self.poppolicyloss.extend(poppolicyloss)
            self._append_loss_diagnostics(
                self.ind_loss_diagnostics,
                getattr(self.rl_alg, 'last_ind_diagnostics', {}),
                len(q1loss),
            )
            self._append_loss_diagnostics(
                self.pop_loss_diagnostics,
                getattr(self.rl_alg, 'last_pop_diagnostics', {}),
                len(q1loss),
            )
            self.epListLoss.extend([self.episode_counter] * len(q1loss))
        elif experience_collected:
            if (0 if self.episode_counter is None else int(self.episode_counter)) < self.training_update_warmup_episodes:
                print(
                    f"Skipping SAC updates for global warmup episode "
                    f"{self.episode_counter + 1}/{self.training_update_warmup_episodes}."
                )
            elif self._is_terrain_prefill_episode():
                print(
                    f"Terrain prefill episode "
                    f"{self.current_training_episode_in_block + 1}/{self.terrain_prefill_episodes} "
                    f"for {self.current_training_terrain}: collecting replay only, no SAC updates."
                )
        self.logTrainLoss() # log data
        self.episode_counter += 1

    def _append_loss_diagnostics(self, target, diagnostics, row_count):
        for key in self.LOSS_DIAGNOSTIC_KEYS:
            value = np.nan
            if diagnostics and key in diagnostics:
                value = diagnostics[key]
            target[key].extend([value] * row_count)

        print(f'episode counter at: {self.episode_counter}')

        self.save_networks()
        return True
      

    def initial_design_loop(self):
        """ The initial training loop for initial designs.

        The initial training loop in which no designs are optimized but only
        initial designs, provided by the environment, are used.

        Args:
            iterations: Integer stating how many training iterations/episodes
                to use per design.

        """
        self.data_design_type = 'Initial'
        params = self.initial_designs[self.design_counter] # choose design based on in which design cycle we are

        SnakeEnv.set_new_design(params)
        self._print_design_installation_prompt(params)
        self.initialize_episode() 
        self._ensure_design_training_schedule()

        
        #for _ in range(self.episode_counter, self.episode_iterations): # train motor controls for this design iteration #added self.episode_counter
        for _ in range(self.episode_counter, self.episode_iterations):
            self.currEp = _
            print('in initial design loop')
            if not self.train_single_iteration():
                print("Stopping initial design loop because no experience was collected.")
                return False

            print(f'range {range(self.episode_counter, self.episode_iterations)}')
        
        # Score this initial design from training replay unless explicit
        # evaluation rollouts were requested.
        self.evaluate_policy()
        self.design_counter += 1
        self.episode_counter = 0
        self.save_networks()

        
        return True
          
    def evaluate_policy(self):
        """Evaluate deterministic policy performance across all terrains.
        Runs repeated deterministic rollouts per terrain and records richer
        return, success, and rollout-length statistics for each design.
        """
        if self.eval_episodes_per_terrain <= 0:
            print(
                "Scoring completed design from training replay. "
                "No extra physical evaluation rollouts will run."
            )
            self._summarize_training_performance()
            return

        policy = self.rl_alg.get_policy_network(self.networks['individual'])
        previous_terrain = SnakeEnv.get_current_terrain()
        eval_design = SnakeEnv.get_current_design()
        total_eval_rollouts = len(self.terrain_sequence) * self.eval_episodes_per_terrain

        print('')
        print('=== EVALUATION START ===')
        print(f'Design cycle: {self.design_counter}')
        print(f'Current design: {self._format_design_for_terminal(eval_design)}')
        print(f'Terrains: {", ".join(self.terrain_sequence)}')
        print(
            f'Evaluation rollouts: {self.eval_episodes_per_terrain} per terrain '
            f'({total_eval_rollouts} total)'
        )
        print(f'Max steps per rollout: {self._episode_length}')
        print('========================')
        print('')

        terrain_returns = {}
        terrain_lengths = {}
        terrain_successes = {}
        terrain_success_steps = {}
        terrain_rollout_seeds = {}
        terrain_reset_failures = {}
        terrain_motor_faults = {}
        terrain_total_rollouts = {}
        terrain_valid_rollouts = {}

        def _finite_values(values):
            arr = np.asarray(values, dtype=np.float32)
            if arr.size == 0:
                return arr
            return arr[np.isfinite(arr)]

        def _json_safe(obj):
            if isinstance(obj, dict):
                return {key: _json_safe(value) for key, value in obj.items()}
            if isinstance(obj, list):
                return [_json_safe(value) for value in obj]
            if isinstance(obj, tuple):
                return [_json_safe(value) for value in obj]
            if isinstance(obj, np.ndarray):
                return [_json_safe(value) for value in obj.tolist()]
            if isinstance(obj, np.generic):
                obj = obj.item()
            if isinstance(obj, float) and not np.isfinite(obj):
                return None
            return obj

        for terrain_order_idx, terrain in enumerate(self.terrain_sequence):
            terrain_idx = self.terrain_name_to_id[terrain]
            SnakeEnv.set_current_terrain(terrain)
            episode_returns = []
            episode_lengths = []
            episode_successes = []
            success_steps = []
            rollout_seeds = []
            reset_failures = 0
            motor_faults = 0
            valid_rollouts = 0

            for rollout_idx in range(self.eval_episodes_per_terrain):
                rollout_number = (
                    terrain_order_idx * self.eval_episodes_per_terrain
                    + rollout_idx
                    + 1
                )
                remaining_rollouts = total_eval_rollouts - rollout_number
                reset_prompt = (
                    "\n"
                    "=== EVALUATION RESET ===\n"
                    f"Design cycle: {self.design_counter}\n"
                    f"Current design: {self._format_design_for_terminal(eval_design)}\n"
                    f"Terrain: {terrain} ({terrain_order_idx + 1}/{len(self.terrain_sequence)})\n"
                    f"Evaluation rollout on this terrain: {rollout_idx + 1}/{self.eval_episodes_per_terrain}\n"
                    f"Total evaluation rollout: {rollout_number}/{total_eval_rollouts}\n"
                    f"Remaining after this rollout: {remaining_rollouts}\n"
                    f"Max steps this rollout: {self._episode_length}\n"
                    f"Put the robot on {terrain}, reset it by hand, then press Enter to start this evaluation rollout."
                )
                print(reset_prompt)
                eval_seed = self._stable_seed(
                    'eval_rollout',
                    self.design_counter,
                    terrain_idx,
                    rollout_idx,
                )
                rollout_seeds.append(int(eval_seed))
                state, _ = self._reset_env_with_motor_recovery(
                    seed=eval_seed,
                    phase=f"evaluation terrain {terrain} rollout {rollout_idx}",
                    options={'reset_prompt': reset_prompt},
                )
                if state is None:
                    print(
                        f"Skipping evaluation rollout {rollout_idx} on terrain '{terrain}' "
                        "because reset recovery did not succeed."
                    )
                    self._disable_motor_torque_with_recovery("evaluation reset failure")
                    episode_returns.append(np.nan)
                    episode_lengths.append(np.nan)
                    episode_successes.append(np.nan)
                    reset_failures += 1
                    continue
                done = False
                steps = 0
                cumulative_reward = 0.0
                success = False
                rollout_faulted = False

                while (not done) and steps < self._episode_length:
                    try:
                        action, _ = policy.get_action(state, deterministic=True)
                    except TypeError:
                        action, _ = policy.get_action(state)

                    try:
                        next_state, reward, terminated, truncated, _ = self.env.step(action)
                    except MotorFaultError as exc:
                        self._recover_motor_fault(
                            f"evaluation terrain {terrain} rollout {rollout_idx} step {steps + 1}",
                            exc,
                        )
                        print("Ending evaluation rollout early after motor recovery.")
                        rollout_faulted = True
                        break
                    cumulative_reward += float(reward)
                    steps += 1
                    success = success or bool(terminated)
                    done = terminated or truncated or (steps >= self._episode_length)
                    state = next_state

                self._disable_motor_torque_with_recovery("end of evaluation rollout")
                if rollout_faulted:
                    print('Evaluation rollout terminated early because a motor fault was detected.')
                    motor_faults += 1
                    episode_returns.append(np.nan)
                    episode_lengths.append(np.nan)
                    episode_successes.append(np.nan)
                else:
                    valid_rollouts += 1
                    episode_returns.append(float(cumulative_reward))
                    episode_lengths.append(int(steps))
                    episode_successes.append(float(success))
                    if success:
                        success_steps.append(int(steps))

            terrain_returns[terrain] = episode_returns
            terrain_lengths[terrain] = episode_lengths
            terrain_successes[terrain] = episode_successes
            terrain_success_steps[terrain] = success_steps
            terrain_rollout_seeds[terrain] = rollout_seeds
            terrain_reset_failures[terrain] = reset_failures
            terrain_motor_faults[terrain] = motor_faults
            terrain_total_rollouts[terrain] = len(episode_returns)
            terrain_valid_rollouts[terrain] = valid_rollouts

        SnakeEnv.set_current_terrain(previous_terrain)

        terrain_means = {}
        terrain_std = {}
        terrain_medians = {}
        terrain_mins = {}
        terrain_success_rates = {
            terrain: float(np.mean(_finite_values(vals))) if len(_finite_values(vals)) > 0 else np.nan
            for terrain, vals in terrain_successes.items()
        }
        terrain_mean_lengths = {
            terrain: float(np.mean(_finite_values(vals))) if len(_finite_values(vals)) > 0 else np.nan
            for terrain, vals in terrain_lengths.items()
        }
        terrain_mean_success_steps = {
            terrain: (float(np.mean(_finite_values(vals))) if len(_finite_values(vals)) > 0 else np.nan)
            for terrain, vals in terrain_success_steps.items()
        }

        for terrain in self.terrain_sequence:
            terrain_eval_returns = _finite_values(terrain_returns[terrain])
            terrain_eval_lengths = _finite_values(terrain_lengths[terrain])
            terrain_eval_successes = _finite_values(terrain_successes[terrain])
            terrain_eval_success_steps = _finite_values(terrain_success_steps[terrain])

            if len(terrain_eval_returns) > 0:
                terrain_means[terrain] = float(np.mean(terrain_eval_returns))
                terrain_std[terrain] = float(np.std(terrain_eval_returns))
                terrain_medians[terrain] = float(np.median(terrain_eval_returns))
                terrain_mins[terrain] = float(np.min(terrain_eval_returns))
            else:
                terrain_means[terrain] = np.nan
                terrain_std[terrain] = np.nan
                terrain_medians[terrain] = np.nan
                terrain_mins[terrain] = np.nan

            terrain_success_rates[terrain] = (
                float(np.mean(terrain_eval_successes)) if len(terrain_eval_successes) > 0 else np.nan
            )
            terrain_mean_lengths[terrain] = (
                float(np.mean(terrain_eval_lengths)) if len(terrain_eval_lengths) > 0 else np.nan
            )
            terrain_mean_success_steps[terrain] = (
                float(np.mean(terrain_eval_success_steps)) if len(terrain_eval_success_steps) > 0 else np.nan
            )

        mean_return_per_terrain = _finite_values(list(terrain_means.values()))
        all_eval_returns = _finite_values([ret for returns in terrain_returns.values() for ret in returns])
        all_eval_lengths = _finite_values([length for lengths in terrain_lengths.values() for length in lengths])
        all_success_steps = _finite_values([step for steps in terrain_success_steps.values() for step in steps])
        valid_terrain_success_rates = _finite_values(list(terrain_success_rates.values()))
        if len(mean_return_per_terrain) > 0:
            mean_return = float(np.mean(mean_return_per_terrain))
            worst_terrain_return = float(np.min(mean_return_per_terrain))
            std_across_terrains = float(np.std(mean_return_per_terrain))
        else:
            mean_return = np.nan
            worst_terrain_return = np.nan
            std_across_terrains = np.nan
        robustness_score = float(mean_return - self.eval_robustness_lambda * std_across_terrains)
        mean_success_rate = float(np.mean(valid_terrain_success_rates)) if len(valid_terrain_success_rates) > 0 else np.nan
        worst_success_rate = float(np.min(valid_terrain_success_rates)) if len(valid_terrain_success_rates) > 0 else np.nan
        overall_median_eval_return = float(np.median(all_eval_returns)) if len(all_eval_returns) > 0 else np.nan
        overall_min_eval_return = float(np.min(all_eval_returns)) if len(all_eval_returns) > 0 else np.nan
        overall_mean_episode_length = float(np.mean(all_eval_lengths)) if len(all_eval_lengths) > 0 else np.nan
        overall_mean_success_steps = (
            float(np.mean(all_success_steps)) if len(all_success_steps) > 0 else np.nan
        )
        total_eval_rollouts = sum(terrain_total_rollouts.values())
        total_valid_eval_rollouts = sum(terrain_valid_rollouts.values())
        total_reset_failures = sum(terrain_reset_failures.values())
        total_motor_faults = sum(terrain_motor_faults.values())
        total_failed_eval_rollouts = total_reset_failures + total_motor_faults
        overall_valid_rollout_rate = (
            float(total_valid_eval_rollouts / total_eval_rollouts) if total_eval_rollouts > 0 else np.nan
        )
        overall_reset_failure_rate = (
            float(total_reset_failures / total_eval_rollouts) if total_eval_rollouts > 0 else np.nan
        )
        overall_motor_fault_rate = (
            float(total_motor_faults / total_eval_rollouts) if total_eval_rollouts > 0 else np.nan
        )
        overall_failure_rate = (
            float(total_failed_eval_rollouts / total_eval_rollouts) if total_eval_rollouts > 0 else np.nan
        )

        current_design = SnakeEnv.get_current_design()
        summary_row = {
            'Date': self.date,
            'Experiment_Seed': int(self.seed),
            'Training_Schedule_Seed': int(self.current_schedule_seed) if self.current_schedule_seed is not None else None,
            'Design_Counter': int(self.design_counter),
            'Episode_Counter': int(self.episode_counter),
            'Training_Episodes_Per_Design': int(self.episode_iterations),
            'Training_Terrain_Block_Size': int(self.training_terrain_block_size),
            'Training_Terrain_Block_Order': '|'.join(self.training_terrain_block_order),
            'Eval_Episodes_Per_Terrain': int(self.eval_episodes_per_terrain),
            'Overall_Total_Eval_Rollouts': int(total_eval_rollouts),
            'Overall_Valid_Eval_Rollouts': int(total_valid_eval_rollouts),
            'Overall_Reset_Failures': int(total_reset_failures),
            'Overall_Motor_Faults': int(total_motor_faults),
            'Overall_Failed_Eval_Rollouts': int(total_failed_eval_rollouts),
            'Overall_Valid_Rollout_Rate': overall_valid_rollout_rate,
            'Overall_Reset_Failure_Rate': overall_reset_failure_rate,
            'Overall_Motor_Fault_Rate': overall_motor_fault_rate,
            'Overall_Failure_Rate': overall_failure_rate,
            'Mean_Return': mean_return,
            'Std_Across_Terrains': std_across_terrains,
            'Worst_Terrain_Return': worst_terrain_return,
            'Mean_Success_Rate': mean_success_rate,
            'Worst_Terrain_Success_Rate': worst_success_rate,
            'Overall_Median_Eval_Return': overall_median_eval_return,
            'Overall_Min_Eval_Return': overall_min_eval_return,
            'Overall_Mean_Episode_Length': overall_mean_episode_length,
            'Overall_Mean_Success_Steps': overall_mean_success_steps,
            'Overall_Mean_Return': mean_return,
            'Std_Across_Terrain_Means': std_across_terrains,
            'Robustness_Lambda': float(self.eval_robustness_lambda),
            'Robustness_Score': robustness_score,
        }
        summary_row.update(self._scale_design_summary_fields(current_design))

        for terrain in self.terrain_sequence:
            total_rollouts = terrain_total_rollouts[terrain]
            valid_rollouts = terrain_valid_rollouts[terrain]
            reset_failures = terrain_reset_failures[terrain]
            motor_faults = terrain_motor_faults[terrain]
            summary_row[f'{terrain}_Total_Rollouts'] = int(total_rollouts)
            summary_row[f'{terrain}_Valid_Rollouts'] = int(valid_rollouts)
            summary_row[f'{terrain}_Reset_Failures'] = int(reset_failures)
            summary_row[f'{terrain}_Motor_Faults'] = int(motor_faults)
            summary_row[f'{terrain}_Valid_Rollout_Rate'] = (
                float(valid_rollouts / total_rollouts) if total_rollouts > 0 else np.nan
            )
            summary_row[f'{terrain}_Reset_Failure_Rate'] = (
                float(reset_failures / total_rollouts) if total_rollouts > 0 else np.nan
            )
            summary_row[f'{terrain}_Motor_Fault_Rate'] = (
                float(motor_faults / total_rollouts) if total_rollouts > 0 else np.nan
            )
            summary_row[f'{terrain}_Mean_Return'] = terrain_means[terrain]
            summary_row[f'{terrain}_Std_Return'] = terrain_std[terrain]
            summary_row[f'{terrain}_Median_Return'] = terrain_medians[terrain]
            summary_row[f'{terrain}_Min_Return'] = terrain_mins[terrain]
            summary_row[f'{terrain}_Success_Rate'] = terrain_success_rates[terrain]
            summary_row[f'{terrain}_Mean_Episode_Length'] = terrain_mean_lengths[terrain]
            summary_row[f'{terrain}_Mean_Success_Steps'] = terrain_mean_success_steps[terrain]

        results_dir = self._checkpoint_results_dir()
        os.makedirs(results_dir, exist_ok=True)

        summary_csv_path = os.path.join(results_dir, f'{self.date}_design_eval_summary.csv')
        summary_df = pd.DataFrame([summary_row])
        if os.path.isfile(summary_csv_path):
            summary_df.to_csv(summary_csv_path, mode='a', header=False, index=False)
        else:
            summary_df.to_csv(summary_csv_path, index=False)

        detail_payload = {
            'summary': _json_safe(summary_row),
            'terrain_episode_returns': terrain_returns,
            'terrain_episode_lengths': terrain_lengths,
            'terrain_episode_successes': terrain_successes,
            'terrain_success_steps': terrain_success_steps,
            'terrain_rollout_seeds': terrain_rollout_seeds,
        }
        detail_json_path = os.path.join(
            results_dir,
            f'{self.date}_DesignCycle{self.design_counter}_ep{self.episode_counter}_eval_summary.json'
        )
        with open(detail_json_path, 'w') as f:
            json.dump(_json_safe(detail_payload), f, indent=2, allow_nan=False)

        print('Evaluation summary:', summary_row)
       
    def save_networks(self):
        """ Saves the networks on the disk.
        """
         # TODO: Edit this to store more efficiently

        results_dir = self._checkpoint_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        checkpoint_prefix = f'{self.date}_DesignCycle{self.design_counter}_ep{self.episode_counter}'

        torch.save(self.rl_alg._ind_policy, os.path.join(results_dir, f'ind_policy_{checkpoint_prefix}_{self.results_tag}.pt'))
        torch.save(self.rl_alg._ind_qf1, os.path.join(results_dir, f'ind_qf1_{checkpoint_prefix}_{self.results_tag}.pt'))
        torch.save(self.rl_alg._ind_qf2, os.path.join(results_dir, f'ind_qf2_{checkpoint_prefix}_{self.results_tag}.pt'))
        torch.save(self.rl_alg._ind_qf1_target, os.path.join(results_dir, f'ind_qf1_tar_{checkpoint_prefix}_{self.results_tag}.pt'))
        torch.save(self.rl_alg._ind_qf2_target, os.path.join(results_dir, f'ind_qf2_tar_{checkpoint_prefix}_{self.results_tag}.pt'))


        torch.save(self.rl_alg._pop_policy, os.path.join(results_dir, f'pop_policy_{checkpoint_prefix}_{self.results_tag}.pt'))
        torch.save(self.rl_alg._pop_qf1, os.path.join(results_dir, f'pop_qf1_{checkpoint_prefix}_{self.results_tag}.pt'))
        torch.save(self.rl_alg._pop_qf2, os.path.join(results_dir, f'pop_qf2_{checkpoint_prefix}_{self.results_tag}.pt'))
        torch.save(self.rl_alg._pop_qf1_target, os.path.join(results_dir, f'pop_qf1_tar_{checkpoint_prefix}_{self.results_tag}.pt'))
        torch.save(self.rl_alg._pop_qf2_target, os.path.join(results_dir, f'pop_qf2_tar_{checkpoint_prefix}_{self.results_tag}.pt'))

        optimized_params = getattr(self, 'optimized_params', None)
        if optimized_params is not None:
            optimized_params = SnakeEnv._coerce_design_vector(optimized_params)

        matched_experiment_settings = {
            'SNAKE_EXPERIMENT_SEED': str(int(self.seed)),
            'SNAKE_ACTIVE_TERRAINS': ','.join(self.terrain_sequence),
            'SNAKE_EPISODES_PER_TERRAIN': str(int(self.training_terrain_block_size)),
            'SNAKE_EVAL_EPISODES_PER_TERRAIN': str(int(self.eval_episodes_per_terrain)),
            'SNAKE_DESIGNS_PER_RUN': os.getenv('SNAKE_DESIGNS_PER_RUN', 'all'),
            'SNAKE_RANDOMIZE_TERRAIN_ORDER': '1' if self.randomize_terrain_order else '0',
        }

        metadata = {
            'seed': int(self.seed),
            'matched_experiment_settings': matched_experiment_settings,
            'results_tag': self.results_tag,
            'design_counter': self.design_counter,
            'episode_counter': self.episode_counter,
            'optimized_params': optimized_params,
            'scale_design_schema_version': int(SnakeEnv.scale_design_schema_version),
            'scale_design_mode': self.design_mode,
            'scale_parameter_names': list(SnakeEnv.design_parameter_names),
            'scale_parameter_bounds': [list(bounds) for bounds in SnakeEnv.design_parameter_bounds],
            'scale_module_length': float(SnakeEnv.scale_module_length),
            'scales_per_module': int(SnakeEnv.scales_per_module),
            'active_terrains': list(self.terrain_sequence),
            'episodes_per_terrain': int(self.training_terrain_block_size),
            'observation_design_features': list(SnakeEnv.get_design_feature_labels()),
            'observation_dim': int(np.prod(self.env.observation_space.shape)),
            'action_dim': int(np.prod(self.env.action_space.shape)),
            'terrain_sequence': list(self.terrain_sequence),
            'terrain_name_to_id': dict(self.terrain_name_to_id),
            'training_terrain_block_order': list(self.training_terrain_block_order),
            'training_episode_schedule': list(self.training_episode_schedule),
            'training_schedule_design_counter': self.training_schedule_design_counter,
            'training_terrain_block_size': self.training_terrain_block_size,
            'randomize_terrain_order': bool(self.randomize_terrain_order),
            'training_schedule_seed': self.current_schedule_seed,
            'policy_action_warmup_episodes': self.policy_action_warmup_episodes,
            'training_update_warmup_episodes': self.training_update_warmup_episodes,
            'eval_episodes_per_terrain': self.eval_episodes_per_terrain,
            'training_score_last_episodes_per_terrain': self.training_score_last_episodes_per_terrain,
            'random_action_prob_start': self.random_action_prob_start,
            'random_action_prob_decay': self.random_action_prob_decay,
            'random_action_prob_min': self.random_action_prob_min,
            'population_training_start_design': self.population_training_start_design,
            'terrain_prefill_episodes': self.terrain_prefill_episodes,
            'action_noise_std': self.action_noise_std,
            'repeat_action_eps': self.repeat_action_eps,
            'repeat_action_perturb_std': self.repeat_action_perturb_std,
            'max_motor_fault_step_retries': self.max_motor_fault_step_retries,
            'motor_fault_step_retry_delay_s': self.motor_fault_step_retry_delay_s,
            'sac_batch_size': int(self.rl_alg._batch_size),
            'individual_sac_updates': int(self.rl_alg._nmbr_ind_updates),
            'population_sac_updates': int(self.rl_alg._nmbr_pop_updates),
            'individual_policy_lr': self.rl_alg._ind_sac_trainer_kwargs['policy_lr'],
            'individual_qf_lr': self.rl_alg._ind_sac_trainer_kwargs['qf_lr'],
            'population_policy_lr': self.rl_alg._pop_sac_trainer_kwargs['policy_lr'],
            'population_qf_lr': self.rl_alg._pop_sac_trainer_kwargs['qf_lr'],
            'individual_batch_fraction': self.replay._individual_batch_fraction,
            'use_legacy_policy_warm_start': self.use_legacy_policy_warm_start,
            'legacy_checkpoint_prefix': None,
            'design_slot_names': list(SnakeEnv.design_slot_names),
        }

        with open(os.path.join(results_dir, f'{checkpoint_prefix}_metadata_{self.results_tag}.json'), 'w') as f:
            json.dump(metadata, f)

        self.save_replay(os.path.join(results_dir, f'replay_{self.date}_DesignCycle{self.design_counter}_{self.results_tag}.pt'))

        print(f"saved networks for design cycle {self.design_counter} and episode {self.episode_counter}")
        print(f"checkpoint prefix: {checkpoint_prefix}")

    def load_networks(self, base_path, checkpoint_prefix):
        metadata_path = self._resolve_tagged_path(base_path, f'{checkpoint_prefix}_metadata', 'json')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"No metadata file found for checkpoint prefix '{checkpoint_prefix}'. "
                "Resume requires metadata so the correct design cycle, episode, "
                "terrain schedule, and optimized design can be restored."
            )

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        if 'design_parameter_options' in metadata:
            raise RuntimeError(
                "This checkpoint uses the old categorical scale-placement schema. "
                "It is incompatible with continuous scale-parameter training. "
                "Start fresh or resume a checkpoint created after the scale-parameter conversion."
            )

        schema_version = int(metadata.get('scale_design_schema_version', -1))
        if schema_version != int(SnakeEnv.scale_design_schema_version):
            raise RuntimeError(
                f"Checkpoint schema version {schema_version} is incompatible with "
                f"current schema version {SnakeEnv.scale_design_schema_version}."
            )

        expected_obs_dim = int(np.prod(self.env.observation_space.shape))
        saved_obs_dim = int(metadata.get('observation_dim', -1))
        if saved_obs_dim != expected_obs_dim:
            raise RuntimeError(
                f"Checkpoint observation_dim={saved_obs_dim} does not match "
                f"current observation_dim={expected_obs_dim}."
            )

        expected_action_dim = int(np.prod(self.env.action_space.shape))
        saved_action_dim = int(metadata.get('action_dim', -1))
        if saved_action_dim != expected_action_dim:
            raise RuntimeError(
                f"Checkpoint action_dim={saved_action_dim} does not match "
                f"current action_dim={expected_action_dim}."
            )

        self.rl_alg._ind_policy.load_state_dict(self._load_trusted_checkpoint(
            self._resolve_tagged_path(base_path, f'ind_policy_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._ind_qf1.load_state_dict(self._load_trusted_checkpoint(
            self._resolve_tagged_path(base_path, f'ind_qf1_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._ind_qf2.load_state_dict(self._load_trusted_checkpoint(
            self._resolve_tagged_path(base_path, f'ind_qf2_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._ind_qf1_target.load_state_dict(self._load_trusted_checkpoint(
            self._resolve_tagged_path(base_path, f'ind_qf1_tar_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._ind_qf2_target.load_state_dict(self._load_trusted_checkpoint(
            self._resolve_tagged_path(base_path, f'ind_qf2_tar_{checkpoint_prefix}', 'pt')
        ).state_dict())

        self.rl_alg._pop_policy.load_state_dict(self._load_trusted_checkpoint(
            self._resolve_tagged_path(base_path, f'pop_policy_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._pop_qf1.load_state_dict(self._load_trusted_checkpoint(
            self._resolve_tagged_path(base_path, f'pop_qf1_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._pop_qf2.load_state_dict(self._load_trusted_checkpoint(
            self._resolve_tagged_path(base_path, f'pop_qf2_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._pop_qf1_target.load_state_dict(self._load_trusted_checkpoint(
            self._resolve_tagged_path(base_path, f'pop_qf1_tar_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._pop_qf2_target.load_state_dict(self._load_trusted_checkpoint(
            self._resolve_tagged_path(base_path, f'pop_qf2_tar_{checkpoint_prefix}', 'pt')
        ).state_dict())

        print(f"loaded networks from checkpoint: {checkpoint_prefix}")

        if metadata:
            self.seed = int(metadata.get('seed', self.seed))
            self.design_counter = metadata['design_counter']
            self.episode_counter = metadata['episode_counter']
            self.optimized_params = metadata.get('optimized_params', None)
            if self.optimized_params is not None:
                self.optimized_params = SnakeEnv._coerce_design_vector(self.optimized_params)
            self.design_mode = metadata.get('scale_design_mode', self.design_mode).strip().lower()
            self.initial_designs = SnakeEnv.get_init_design_parameters(self.design_mode)
            self.num_init_designs = len(self.initial_designs)

            saved_terrain_sequence = metadata.get(
                'active_terrains',
                metadata.get('terrain_sequence', self.terrain_sequence),
            )
            saved_terrain_sequence = self._parse_active_terrains(','.join(saved_terrain_sequence))
            saved_training_block_size = int(
                metadata.get(
                    'episodes_per_terrain',
                    metadata.get('training_terrain_block_size', self.training_terrain_block_size),
                )
            )
            self.randomize_terrain_order = self._read_bool_env(
                'SNAKE_RANDOMIZE_TERRAIN_ORDER',
                default=bool(metadata.get('randomize_terrain_order', self.randomize_terrain_order)),
            )

            requested_terrain_sequence = saved_terrain_sequence
            if os.getenv('SNAKE_ACTIVE_TERRAINS') is not None:
                requested_terrain_sequence = self._parse_active_terrains()
            requested_training_block_size = saved_training_block_size
            if os.getenv('SNAKE_EPISODES_PER_TERRAIN') is not None:
                requested_training_block_size = max(1, int(os.getenv('SNAKE_EPISODES_PER_TERRAIN')))

            terrain_changed = requested_terrain_sequence != saved_terrain_sequence
            block_size_changed = requested_training_block_size != saved_training_block_size
            if (terrain_changed or block_size_changed) and int(self.episode_counter) != 0:
                raise RuntimeError(
                    "Changing SNAKE_ACTIVE_TERRAINS or SNAKE_EPISODES_PER_TERRAIN during "
                    "a design is not supported. Resume at a design boundary where "
                    "episode_counter == 0, or keep the checkpoint terrain settings."
                )

            self.terrain_sequence = requested_terrain_sequence
            self.training_terrain_block_size = requested_training_block_size
            saved_results_tag = metadata.get('results_tag')
            if saved_results_tag and os.getenv('SNAKE_ACTIVE_TERRAINS') is None and os.getenv('SNAKE_RESULTS_TAG') is None:
                self.results_tag = saved_results_tag
            self.legacy_results_tags = list(dict.fromkeys(
                tag for tag in ([self.results_tag, saved_results_tag] + self.terrain_sequence + ['carton'])
                if tag
            ))
            if terrain_changed or block_size_changed:
                self.training_terrain_block_order = []
                self.training_episode_schedule = []
                self.training_schedule_design_counter = None
                self.current_schedule_seed = None
            else:
                self.training_terrain_block_order = metadata.get('training_terrain_block_order', [])
                self.training_episode_schedule = metadata.get('training_episode_schedule', [])
                self.training_schedule_design_counter = metadata.get(
                    'training_schedule_design_counter',
                    self.design_counter if self.training_episode_schedule else None,
                )
            self.current_schedule_seed = (
                None if (terrain_changed or block_size_changed)
                else metadata.get('training_schedule_seed', self.current_schedule_seed)
            )
            self.policy_action_warmup_episodes = metadata.get(
                'policy_action_warmup_episodes',
                self.policy_action_warmup_episodes,
            )
            self.training_update_warmup_episodes = metadata.get(
                'training_update_warmup_episodes',
                self.training_update_warmup_episodes,
            )
            self.eval_episodes_per_terrain = max(
                0,
                int(
                    os.getenv(
                        'SNAKE_EVAL_EPISODES_PER_TERRAIN',
                        self.eval_episodes_per_terrain,
                    )
                )
            )
            self.training_score_last_episodes_per_terrain = max(
                1,
                int(
                    os.getenv(
                        'SNAKE_TRAINING_SCORE_LAST_EPISODES_PER_TERRAIN',
                        metadata.get(
                            'training_score_last_episodes_per_terrain',
                            self.training_score_last_episodes_per_terrain,
                        ),
                    )
                )
            )
            self.random_action_prob_start = metadata.get(
                'random_action_prob_start',
                self.random_action_prob_start,
            )
            self.random_action_prob_decay = metadata.get(
                'random_action_prob_decay',
                self.random_action_prob_decay,
            )
            self.random_action_prob_min = metadata.get(
                'random_action_prob_min',
                self.random_action_prob_min,
            )
            self.max_motor_fault_step_retries = max(
                0,
                int(
                    os.getenv(
                        'SNAKE_MOTOR_FAULT_STEP_RETRIES',
                        metadata.get(
                            'max_motor_fault_step_retries',
                            self.max_motor_fault_step_retries,
                        ),
                    )
                )
            )
            self.motor_fault_step_retry_delay_s = max(
                0.0,
                float(
                    os.getenv(
                        'SNAKE_MOTOR_FAULT_STEP_RETRY_DELAY_S',
                        metadata.get(
                            'motor_fault_step_retry_delay_s',
                            self.motor_fault_step_retry_delay_s,
                        ),
                    )
                )
            )
            if metadata.get('use_legacy_policy_warm_start', False):
                raise RuntimeError(
                    "This checkpoint was saved from a legacy warm-start run and is not "
                    "compatible with continuous scale-parameter training."
                )
            self.use_legacy_policy_warm_start = False
            default_pop_train_start_design = '0'
            default_terrain_prefill_episodes = '0'
            default_action_noise_std = '0.02'
            default_repeat_action_eps = '0.02'
            default_repeat_action_perturb_std = '0.02'
            default_policy_warmup_episodes = str(self.policy_action_warmup_episodes)
            default_update_warmup_episodes = str(self.training_update_warmup_episodes)
            default_random_action_prob_start = str(self.random_action_prob_start)
            default_random_action_prob_decay = str(self.random_action_prob_decay)
            default_random_action_prob_min = str(self.random_action_prob_min)
            self.population_training_start_design = int(
                os.getenv('SNAKE_POP_TRAIN_START_DESIGN', default_pop_train_start_design)
            )
            self.terrain_prefill_episodes = max(
                0,
                int(os.getenv('SNAKE_TERRAIN_PREFILL_EPISODES', default_terrain_prefill_episodes))
            )
            self.policy_action_warmup_episodes = max(
                0,
                int(os.getenv('SNAKE_POLICY_WARMUP_EPISODES', default_policy_warmup_episodes)),
            )
            self.training_update_warmup_episodes = max(
                0,
                int(os.getenv('SNAKE_UPDATE_WARMUP_EPISODES', default_update_warmup_episodes)),
            )
            self.random_action_prob_start = max(
                0.0,
                float(os.getenv('SNAKE_RANDOM_ACTION_PROB_START', default_random_action_prob_start)),
            )
            self.random_action_prob_decay = max(
                0.0,
                float(os.getenv('SNAKE_RANDOM_ACTION_PROB_DECAY', default_random_action_prob_decay)),
            )
            self.random_action_prob_min = max(
                0.0,
                float(os.getenv('SNAKE_RANDOM_ACTION_PROB_MIN', default_random_action_prob_min)),
            )
            self.action_noise_std = max(
                0.0,
                float(os.getenv('SNAKE_ACTION_NOISE_STD', default_action_noise_std)),
            )
            self.repeat_action_eps = max(
                0.0,
                float(os.getenv('SNAKE_REPEAT_ACTION_EPS', default_repeat_action_eps)),
            )
            self.repeat_action_perturb_std = max(
                0.0,
                float(os.getenv('SNAKE_REPEAT_ACTION_PERTURB_STD', default_repeat_action_perturb_std)),
            )
            self.episode_iterations = len(self.terrain_sequence) * self.training_terrain_block_size
            self._seed_global_rngs('resume', self.design_counter, self.episode_counter)
            print(f"restored design_counter={self.design_counter}, episode_counter={self.episode_counter}")
            print(
                "resume training knobs -> "
                f"population start design: {self.population_training_start_design}, "
                f"terrain prefill episodes: {self.terrain_prefill_episodes}, "
                f"policy warmup episodes: {self.policy_action_warmup_episodes}, "
                f"update warmup episodes: {self.training_update_warmup_episodes}, "
                f"random action start/min: {self.random_action_prob_start:.3f}/{self.random_action_prob_min:.3f}, "
                f"action_noise_std: {self.action_noise_std:.3f}, "
                f"repeat_action_perturb_std: {self.repeat_action_perturb_std:.3f}"
            )
        else:
            raise FileNotFoundError(
                f"No metadata file found for checkpoint prefix '{checkpoint_prefix}'. "
                "Resume requires metadata so the correct design cycle, episode, "
                "terrain schedule, and optimized design can be restored."
            )

        replay_path = self._resolve_tagged_path(base_path, f'replay_{checkpoint_prefix.split("_ep")[0]}', 'pt')
        if os.path.exists(replay_path):
            if not self.load_replay(replay_path):
                raise RuntimeError(f"Replay buffer could not be restored from {replay_path}.")
            self._refresh_active_terrain_filter()
            print("Replay contains", self.replay._individual_buffer._size, "steps")
        else:
            raise FileNotFoundError(
                f"No replay buffer found for checkpoint prefix '{checkpoint_prefix}'. "
                "Resume requires replay state so population/design optimization remains consistent."
            )



    def _serialize_replay_buffer(self, buffer):
        active_size = int(buffer._size)
        max_size = int(buffer._max_replay_buffer_size)

        if active_size <= 0:
            active_indices = np.array([], dtype=np.int64)
        elif active_size < max_size and buffer._top == active_size:
            active_indices = np.arange(active_size, dtype=np.int64)
        else:
            start = (buffer._top - active_size) % max_size
            if start < buffer._top:
                active_indices = np.arange(start, buffer._top, dtype=np.int64)
            else:
                active_indices = np.concatenate([
                    np.arange(start, max_size, dtype=np.int64),
                    np.arange(0, buffer._top, dtype=np.int64),
                ])

        return {
            "observations": buffer._observations[active_indices].copy(),
            "actions": buffer._actions[active_indices].copy(),
            "rewards": buffer._rewards[active_indices].copy(),
            "terminals": buffer._terminals[active_indices].copy(),
            "next_observations": buffer._next_obs[active_indices].copy(),
            "env_infos": {
                key: value[active_indices].copy()
                for key, value in buffer._env_infos.items()
            },
            "env_info_keys": list(buffer._env_info_keys),
            "_top": active_size % max_size if max_size else 0,
            "_size": active_size,
            "_replace": buffer._replace,
            "_max_replay_buffer_size": max_size,
            "_observation_dim": buffer._observation_dim,
            "_action_dim": buffer._action_dim,
        }

    def _restore_replay_buffer(self, buffer, data):
        max_size = int(data.get("_max_replay_buffer_size", buffer._max_replay_buffer_size))
        size = int(data.get("_size", 0))
        saved_observation_dim = data.get("_observation_dim")
        saved_action_dim = data.get("_action_dim")
        if saved_observation_dim is None and "observations" in data:
            saved_observation_shape = np.shape(data["observations"])
            if len(saved_observation_shape) > 1:
                saved_observation_dim = saved_observation_shape[1]
        if saved_action_dim is None and "actions" in data:
            saved_action_shape = np.shape(data["actions"])
            if len(saved_action_shape) > 1:
                saved_action_dim = saved_action_shape[1]

        saved_observation_dim = int(saved_observation_dim or buffer._observation_dim)
        saved_action_dim = int(saved_action_dim or buffer._action_dim)
        if saved_observation_dim != buffer._observation_dim or saved_action_dim != buffer._action_dim:
            raise ValueError(
                "Replay buffer shape mismatch: "
                f"saved obs/action dims {saved_observation_dim}/{saved_action_dim}, "
                f"current obs/action dims {buffer._observation_dim}/{buffer._action_dim}. "
                "Start with a fresh replay buffer or migrate the saved replay before resuming."
            )

        buffer._observations = np.zeros((max_size, buffer._observation_dim))
        buffer._actions = np.zeros((max_size, buffer._action_dim))
        buffer._rewards = np.zeros((max_size, 1))
        buffer._terminals = np.zeros((max_size, 1), dtype='uint8')
        buffer._next_obs = np.zeros((max_size, buffer._observation_dim))

        saved_env_infos = data.get("env_infos", {})
        buffer._env_info_keys = data.get("env_info_keys", list(saved_env_infos.keys()))
        buffer._env_infos = {}
        for key in buffer._env_info_keys:
            saved_values = saved_env_infos.get(key)
            if saved_values is None:
                buffer._env_infos[key] = np.zeros((max_size, 1))
            else:
                width = saved_values.shape[1] if saved_values.ndim > 1 else 1
                buffer._env_infos[key] = np.zeros((max_size, width), dtype=saved_values.dtype)

        if size > 0:
            buffer._observations[:size] = data["observations"][:size]
            buffer._actions[:size] = data["actions"][:size]
            buffer._rewards[:size] = data["rewards"][:size]
            buffer._terminals[:size] = data["terminals"][:size]
            buffer._next_obs[:size] = data["next_observations"][:size]
            for key in buffer._env_info_keys:
                saved_values = saved_env_infos.get(key)
                if saved_values is not None:
                    buffer._env_infos[key][:size] = saved_values[:size]

        buffer._top = size % max_size if max_size else 0
        buffer._size = size
        buffer._replace = data.get("_replace", buffer._replace)
        buffer._max_replay_buffer_size = max_size
        buffer._observation_dim = saved_observation_dim
        buffer._action_dim = saved_action_dim

    def save_replay(self, filepath):
        """Save full coadaptation replay state to disk."""

        try:
            data = {
                "version": 2,
                "mode": self.replay._mode,
                "ep_counter": self.replay._ep_counter,
                "expect_init_state": self.replay._expect_init_state,
                "individual_buffer": self._serialize_replay_buffer(self.replay._individual_buffer),
                "population_buffer": self._serialize_replay_buffer(self.replay._population_buffer),
                "init_state_buffer": self._serialize_replay_buffer(self.replay._init_state_buffer),
            }
            torch.save(data, filepath)
            print(
                "saved replay buffers to {} (individual={}, population={}, init={})".format(
                    filepath,
                    self.replay._individual_buffer._size,
                    self.replay._population_buffer._size,
                    self.replay._init_state_buffer._size,
                )
            )
        except Exception as e:
            print(f"failed to save replay buffer: {e}")

    def load_replay(self, filepath):
        """Load full coadaptation replay state from disk."""
        try:
            data = self._load_trusted_checkpoint(filepath)

            if "individual_buffer" not in data:
                # Backward-compatible fallback for older checkpoints that only
                # stored one replay buffer. Mirror it into all buffers so
                # resumed training and PSO can still sample population/start data.
                self._restore_replay_buffer(self.replay._individual_buffer, data)
                self._restore_replay_buffer(self.replay._population_buffer, data)
                self._restore_replay_buffer(self.replay._init_state_buffer, data)
                self.replay._mode = data.get("mode", self.replay._mode)
                self.replay._ep_counter = data.get("ep_counter", self.replay._ep_counter)
                self.replay._expect_init_state = data.get("expect_init_state", self.replay._expect_init_state)
                print(
                    "loaded legacy replay buffer from {} (individual={}, population={}, init={})".format(
                        filepath,
                        self.replay._individual_buffer._size,
                        self.replay._population_buffer._size,
                        self.replay._init_state_buffer._size,
                    )
                )
                return True

            self._restore_replay_buffer(self.replay._individual_buffer, data["individual_buffer"])
            self._restore_replay_buffer(self.replay._population_buffer, data["population_buffer"])
            self._restore_replay_buffer(self.replay._init_state_buffer, data["init_state_buffer"])
            self.replay._mode = data.get("mode", self.replay._mode)
            self.replay._ep_counter = data.get("ep_counter", self.replay._ep_counter)
            self.replay._expect_init_state = data.get("expect_init_state", self.replay._expect_init_state)

            print(
                "loaded replay buffers from {} (individual={}, population={}, init={})".format(
                    filepath,
                    self.replay._individual_buffer._size,
                    self.replay._population_buffer._size,
                    self.replay._init_state_buffer._size,
                )
            )
            return True
        except Exception as e:
            print(f"failed to load replay buffer: {e}")
            return False

    def logData(self):
        xPositionList, yPositionList = SnakeEnv.returnOptiXList()
        min_len = min(len(self.timesteps), len(xPositionList))

        # trim all lists to the same length
        self.timesteps = self.timesteps[:min_len]
        self.timestepRewards = self.timestepRewards[:min_len]
        self.cumulativeRewards = self.cumulativeRewards[:min_len]
        self.progressRewardComponents = self.progressRewardComponents[:min_len]
        self.distanceProgressCmComponents = self.distanceProgressCmComponents[:min_len]
        self.rawDistanceProgressCmComponents = self.rawDistanceProgressCmComponents[:min_len]
        self.windowProgressCmComponents = self.windowProgressCmComponents[:min_len]
        self.xDriftPenaltyComponents = self.xDriftPenaltyComponents[:min_len]
        self.headingPenaltyComponents = self.headingPenaltyComponents[:min_len]
        self.livingPenaltyComponents = self.livingPenaltyComponents[:min_len]
        self.noProgressPenaltyComponents = self.noProgressPenaltyComponents[:min_len]
        self.backwardPenaltyComponents = self.backwardPenaltyComponents[:min_len]
        xPositionList = xPositionList[-min_len:]
        yPositionList = yPositionList[-min_len:]
        self.epList = self.epList[:min_len]

        for i in range(len(self.actionList)):
            self.actionList[i] = self.actionList[i][:min_len]
        for i in range(len(self.stateList)):
            self.stateList[i] = self.stateList[i][:min_len]
        rewardDF = pd.DataFrame()

        rewardDF['Run_ID'] = [self.run_id] * len(self.timesteps)
        rewardDF['Episode'] = [self.episode_counter]* len(self.timesteps)
        rewardDF['Timestep'] = self.timesteps
        rewardDF['X_Position']= xPositionList # added this, need to see if it works
        rewardDF['Y_Position']= yPositionList # added this, need to see if it works
        rewardDF['Rewards'] = self.timestepRewards
        rewardDF['Cumulative_Rewards'] = self.cumulativeRewards
        rewardDF['Progress_Reward'] = self.progressRewardComponents
        rewardDF['Distance_Progress_Cm'] = self.distanceProgressCmComponents
        rewardDF['Raw_Distance_Progress_Cm'] = self.rawDistanceProgressCmComponents
        rewardDF['Window_Progress_Cm'] = self.windowProgressCmComponents
        rewardDF['X_Drift_Penalty'] = self.xDriftPenaltyComponents
        rewardDF['Heading_Penalty'] = self.headingPenaltyComponents
        rewardDF['Living_Penalty'] = self.livingPenaltyComponents
        rewardDF['No_Progress_Penalty'] = self.noProgressPenaltyComponents
        rewardDF['Backward_Penalty'] = self.backwardPenaltyComponents
        rewardDF['Terrain'] = [SnakeEnv.get_current_terrain()] * len(self.timesteps)
        design = SnakeEnv.get_current_design()
        rewardDF['Experiment_Seed'] = [int(self.seed)] * len(self.timesteps)
        rewardDF['Episode_Seed'] = [self.current_episode_seed] * len(self.timesteps)
        rewardDF['Terrain_ID'] = [self.current_training_terrain_id] * len(self.timesteps)
        rewardDF['Terrain_Block_Index'] = [self.current_training_block_index] * len(self.timesteps)
        rewardDF['Episode_In_Terrain_Block'] = [self.current_training_episode_in_block] * len(self.timesteps)
        design_summary = self._scale_design_summary_fields(design)
        for column_name, value in design_summary.items():
            rewardDF[column_name] = [value] * len(self.timesteps)
        for module in SnakeEnv.expand_design_to_modules(design):
            module_idx = module['module']
            rewardDF[f'Module{module_idx}_Scale_Group'] = [module['group']] * len(self.timesteps)
            rewardDF[f'Module{module_idx}_Width_Ratio'] = [module['width_ratio']] * len(self.timesteps)
            rewardDF[f'Module{module_idx}_Actual_Width'] = [module['actual_width']] * len(self.timesteps)
            rewardDF[f'Module{module_idx}_Attack_Angle_Deg'] = [module['attack_angle_deg']] * len(self.timesteps)

        # log state variablesmotor_and_coadaptation/CoadaptationCode/train_coadapt.py
        for motor_idx, motor_actions in enumerate(self.actionList):
            rewardDF[f'Motor{motor_idx + 1}_Action'] = motor_actions

        observation_labels = SnakeEnv.get_observation_feature_labels()
        for obs_idx, obs_values in enumerate(self.stateList):
            if obs_idx < len(observation_labels):
                column_name = observation_labels[obs_idx]
            else:
                column_name = f'Obs_{obs_idx}'
            rewardDF[column_name] = obs_values

        self._upsert_csv_rows(self.filename, rewardDF, ['Run_ID', 'Episode'])

    def logTrainLoss(self):
        lossDF = pd.DataFrame()
        lossDF['Run_ID'] = [self.run_id] * len(self.epListLoss)
        lossDF['Episode'] = self.epListLoss
        lossDF['Ind_Q1_Loss'] = self.q1loss
        lossDF['Ind_Q2_Loss'] = self.q2loss
        lossDF['Ind_Policy_Loss'] = self.policyloss

        lossDF['Pop_Q1_Loss'] = self.popq1loss
        lossDF['Pop_Q2_Loss'] = self.popq2loss
        lossDF['Pop_Policy_Loss'] = self.poppolicyloss
        for key in self.LOSS_DIAGNOSTIC_KEYS:
            column_suffix = key.replace(' ', '_')
            lossDF[f'Ind_{column_suffix}'] = self.ind_loss_diagnostics[key]
            lossDF[f'Pop_{column_suffix}'] = self.pop_loss_diagnostics[key]
        self._upsert_csv_rows(self.lossFilename, lossDF, ['Run_ID', 'Episode'])



    def passLocks(self, oLock, mLock):
        # pass locks into the environment  
        SnakeEnv.passLocksToEnv(oLock, mLock)
        
    def optiPos(self, stopEvent):
        # to run on thread and interact with snake environment
        while True:   
            SnakeEnv.optiPos()
            if stopEvent.is_set():
                break
        

    def motorPos(self, stopEvent):
        # to run on thread and interact with snake environment
        while True:
            SnakeEnv.motorPos()
            if stopEvent.is_set():
                break
    
    from itertools import tee


if __name__ == '__main__':

    
    gc.collect()
    gc.set_threshold(0)

    startTrainingSession = False
    stopEvent = threading.Event()

    
    stopEvent = threading.Event()
    trainingObj = Train()
    optiLock = threading.Lock()
    motorLock = threading.Lock()
    trainingObj.passLocks(optiLock, motorLock)

    # if resuming from a checkpoint:
    base_path = trainingObj._checkpoint_results_dir()
    checkpoint_prefix = os.getenv("SNAKE_CHECKPOINT_PREFIX")

    # Default to a fresh continuous scale-parameter run. Set
    # SNAKE_RESUME_CHECKPOINT=1 when you intentionally want to continue a
    # saved results_bazyli checkpoint created with the same scale schema.
    resuming_from_checkpoint = trainingObj._read_bool_env("SNAKE_RESUME_CHECKPOINT", default=False)

    if resuming_from_checkpoint:
        if not checkpoint_prefix:
            raise ValueError(
                "SNAKE_RESUME_CHECKPOINT=1 requires SNAKE_CHECKPOINT_PREFIX. "
                "Use the prefix printed in the saved checkpoint name, for example "
                "2026_04_16_DesignCycle1_ep0."
            )
        trainingObj.load_networks(base_path, checkpoint_prefix)
    else:
        trainingObj.episode_counter = 0
        print("Starting fresh: episode_counter set to 0")

    designs_per_run_env = os.getenv('SNAKE_DESIGNS_PER_RUN', '').strip().lower()
    if designs_per_run_env in ('', '0', 'all', 'none'):
        designs_per_run = None
        print('Design cycles per run: all remaining cycles.')
    else:
        designs_per_run = max(1, int(designs_per_run_env))
        print(f'Design cycles per run: {designs_per_run}')

    # run threads as before
    motorThread = threading.Thread(target=trainingObj.motorPos, args=(stopEvent,), daemon=True) 
    optiThread = threading.Thread(target=trainingObj.optiPos, args=(stopEvent,), daemon=True)
    trainingloopThread = threading.Thread(target=trainingObj.run, args=(stopEvent, designs_per_run))

    try:
        motorThread.start()
        optiThread.start()
        trainingloopThread.start()
        trainingloopThread.join()
    finally:
        stopEvent.set()
        try:
            torque_disabled = SnakeEnv.disableMotorTorque()
            if not torque_disabled:
                print("Could not disable motor torque during shutdown; forcing DYNAMIXEL reboot.")
                recovered = SnakeEnv.recoverMotorFault(
                    context="shutdown torque disable failed",
                    force_reboot=True,
                )
                if recovered and not SnakeEnv.disableMotorTorque():
                    print("Motor torque still could not be disabled after shutdown reboot.")
        except Exception as exc:
            print(f"Could not disable motor torque during shutdown: {exc}. Forcing DYNAMIXEL reboot.")
            try:
                recovered = SnakeEnv.recoverMotorFault(
                    context=f"shutdown torque disable raised: {exc}",
                    force_reboot=True,
                )
                if recovered and not SnakeEnv.disableMotorTorque():
                    print("Motor torque still could not be disabled after shutdown reboot.")
            except Exception as recovery_exc:
                print(f"Shutdown motor recovery raised an exception: {recovery_exc}")
        motorThread.join(timeout=2.0)
        optiThread.join(timeout=2.0)
