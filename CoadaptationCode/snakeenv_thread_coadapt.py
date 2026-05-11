import gymnasium
from gymnasium import spaces
import numpy as np
import motorssynced
import optitrack
import threading
import math
import random
import matplotlib.pyplot as plt
import time
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
import os
import sys
import copy
from collections import deque
from datetime import datetime

gymnasium.envs.register(
    id = "SnakeRobot",
    entry_point = f"{__name__}:SnakeEnv",
    max_episode_steps = 250,  # maybe come back and change
    reward_threshold = 1000,
    
)
global optiPos, motorPos

class MotorFaultError(RuntimeError):
    pass

class SnakeEnv(gymnasium.Env):
    # static variables so can be accessed between static and non static methods
    optiPosition = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    motorPosition = []
    optiXTrack = []
    optiYTrack = []
    prevPos = [0,0,0]
    optiRelPos = []
    # Hardware clients are initialized lazily in __init__ to avoid side-effects
    # (serial/network connections) during module import.
    motors = None
    opti = None
    motorLock = threading.Lock()
    optiLock = threading.Lock()
    starting_angle = None

    bla = time.time()
    opti_last_print_time = 0.0
    opti_last_update_time = 0.0
    opti_last_stale_warn_time = 0.0

    '''
       Robot has 7 motors and 8 snake segments
       Action Space: 7
       Observation Space: 4 normalized motion features + 7 normalized motors + 6 continuous scale-design features
    '''

    # setting up design framework
    # Continuous scale-parameter design. The physical layout is fixed to an
    # alternating A/B pattern: A on modules 1,3,5,7 and B on modules 2,4,6,8.
    scale_design_schema_version = 3
    scale_module_length = 60.0
    scales_per_module = 7
    scale_pitch = scale_module_length / scales_per_module
    design_parameter_names = [
        "A_width_ratio",
        "A_attack_angle_deg",
        "B_width_ratio",
        "B_attack_angle_deg",
    ]
    design_parameter_bounds = [
        (0.45, 0.90),
        (0.0, 15.0),
        (0.45, 0.90),
        (0.0, 15.0),
    ]
    design_feature_names = [
        "A_width_norm",
        "A_attack_angle_norm",
        "B_width_norm",
        "B_attack_angle_norm",
        "Delta_width_norm",
        "Delta_attack_angle_norm",
    ]
    design_slot_count = 8
    module_group_pattern = ["A", "B", "A", "B", "A", "B", "A", "B"]
    current_design = [0.63, 0.0, 0.63, 0.0]
    terrains = ['artificial_grass', 'cardboard', 'carpet', 'foam']
    # Terrain IDs are saved in replay buffers; keep existing IDs stable.
    terrain_name_to_id = {
        'carpet': 0,
        'cardboard': 1,
        'artificial_grass': 2,
        'foam': 3,
    }
    terrain_id_to_name = {
        0: 'carpet',
        1: 'cardboard',
        2: 'artificial_grass',
        3: 'foam',
    }
    current_terrain = terrains[0]

    # Initial matched seeds for the homogeneous-vs-heterogeneous experiment.
    homogeneous_init_design_parameters = [
        [0.63, 0.0, 0.63, 0.0],
        [0.63, 15.0, 0.63, 15.0],
        [0.90, 0.0, 0.90, 0.0],
        [0.90, 15.0, 0.90, 15.0],
    ]
    heterogeneous_init_design_parameters = [
        [0.63, 0.0, 0.90, 15.0],
        [0.90, 15.0, 0.63, 0.0],
        [0.63, 15.0, 0.90, 0.0],
        [0.90, 0.0, 0.63, 15.0],
    ]
    init_design_parameters = heterogeneous_init_design_parameters

    design_slot_names = [f'Module{i + 1}' for i in range(design_slot_count)]
    config_numpy = np.asarray([-0.2, -1.0, -0.2, -1.0, 0.0, 0.0], dtype=np.float32)
    base_feature_dim = 11
    design_dims = list(range(base_feature_dim, base_feature_dim + len(config_numpy)))
    print('design dimensions!', design_dims)
    
    def __init__(self):
    
        self.rewardScale = 100 # scale rewards
        self.motorMin = 1422 #1500 #1422
        self.motorMax = 2674 #2500 #2673
        
       
        obs_dim = SnakeEnv.base_feature_dim + len(SnakeEnv.config_numpy)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dim,),
            dtype='float32'
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype='float32')  # normalized actions

        # OptiTrack stream is in meters; this env scales position by *100.
        # Targeting is Z-based for termination/reward.
        # target z = -900 mm -> env units -90.0 (position is meters * 100).
        self.targetDistanceZ = 190.0  # overwritten on reset from observed start z
        self.starting_position_z = None
        self.starting_position_x = None
        self.progress_direction_z = -1
        self.targetPositionY = 19.5231
        self.targetPositionZ = -85.0
        self.x_drift_penalty_start = 15.0
        self.x_drift_penalty_full = 80.0
        self.x_drift_observation_scale = 80.0
        self.progress_filter_alpha = 0.25
        # Keep a tiny deadzone for OptiTrack jitter, but still reward the
        # millimeter-to-centimeter progress this robot makes per control step.
        self.progress_deadzone_cm = 0.05
        self.progress_fullscale_cm = 2.0
        self.progress_window_size = 6
        self.window_progress_threshold_cm = 0.5
        self.no_progress_penalty_max = 0.05
        self.step_living_penalty = 0.02
        self.heading_penalty_deadzone = 25.0 / 180.0
        self.x_drift_penalty_scale = 0.10
        self.heading_penalty_scale = 0.05
        self.terminal_reward_bonus = 5.0
        self.reward_clip_min = -3.0
        self.reward_clip_max = 6.0
        self._interactive_reset_default = self._read_interactive_reset_default()
        self._auto_motor_reset_default = self._read_bool_env(
            'SNAKE_AUTO_MOTOR_RESET',
            default=False,
        )

               
        # init other things
        # moved these class declarations to static
        SnakeEnv._ensure_hardware_initialized()
        self.motors = SnakeEnv.motors
        self.opti = SnakeEnv.opti
        #self.opti.optiTrackInit()

        self.currPosition =[]
        self.newAction = []

        self.reward = 0
        self.prevDist = 0
        self.prevPos = 0 
        self.prevXpos = 0
        self._prev_raw_obs = None
        self.filtered_z = None
        self.filtered_x = None
        self.filtered_heading = None
        self.prev_filtered_distance_to_goal = None
        self.filtered_distance_window = deque(maxlen=self.progress_window_size + 1)

        self.distList = []
        self.rewardList = []
        self.xPosList = []
        self.i = 0

        # data frame for logging data
        self.df = pd.DataFrame(columns=['Action Sent','Opti Position', 'Motor Position','Reward'])

        # set up files
        self.filename = "Training_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        
     
    def _read_bool_env(self, name, default):
        env_value = os.getenv(name)
        if env_value is None:
            return default
        return env_value.strip().lower() in ('1', 'true', 'yes', 'on')

    def _read_interactive_reset_default(self):
        return self._read_bool_env('SNAKE_INTERACTIVE_RESET', sys.stdin.isatty())

    def _should_prompt_for_reset(self, options=None):
        if options and 'interactive_reset' in options:
            return bool(options['interactive_reset'])
        return self._interactive_reset_default

    def _should_auto_motor_reset(self, options=None):
        if options and 'auto_motor_reset' in options:
            return bool(options['auto_motor_reset'])
        return self._auto_motor_reset_default

    def _disable_motor_torque_for_manual_reset(self):
        SnakeEnv.motorLock.acquire()
        try:
            torque_disabled = SnakeEnv.motors.disableTorque()
        finally:
            SnakeEnv.motorLock.release()

        if not torque_disabled:
            print(
                "Disabling motor torque before manual reset reported an issue. "
                "Forcing DYNAMIXEL reboot."
            )
            recovered = SnakeEnv.recoverMotorFault(
                context="manual reset torque disable failed",
                force_reboot=True,
            )
            if not recovered:
                raise MotorFaultError(
                    "Failed to disable motor torque before manual reset. "
                    "Automatic DYNAMIXEL recovery/reboot was attempted."
                )

            SnakeEnv.motorLock.acquire()
            try:
                torque_disabled = SnakeEnv.motors.disableTorque()
            finally:
                SnakeEnv.motorLock.release()

            if not torque_disabled:
                raise MotorFaultError(
                    "Failed to disable motor torque after manual-reset recovery. "
                    "Automatic DYNAMIXEL recovery/reboot was attempted."
                )
        return torque_disabled

    def _refresh_manual_reset_motor_position(self):
        SnakeEnv.motorLock.acquire()
        try:
            SnakeEnv.motorPosition = SnakeEnv.motors.readPos()
        finally:
            SnakeEnv.motorLock.release()

    def _normalize_motor_positions(self, motor_positions):
        motor_positions = np.asarray(motor_positions, dtype=np.float32)
        # MotorsSynced.readPos already returns normalized feedback.
        if motor_positions.size and np.all(np.isfinite(motor_positions)):
            if np.max(np.abs(motor_positions)) <= 1.0:
                return np.clip(motor_positions, -1.0, 1.0)

        motor_span = max(self.motorMax - self.motorMin, 1)
        normalized = 2.0 * (motor_positions - self.motorMin) / motor_span - 1.0
        return np.clip(normalized, -1.0, 1.0)

    def _compute_x_drift(self, x_abs):
        if self.starting_position_x is None:
            return 0.0, 0.0, 0.0, 0.0

        signed_x_drift = float(x_abs - self.starting_position_x)
        abs_x_drift = abs(signed_x_drift)
        x_drift_norm = float(
            np.clip(signed_x_drift / max(self.x_drift_observation_scale, 1.0), -1.0, 1.0)
        )

        penalty_span = max(self.x_drift_penalty_full - self.x_drift_penalty_start, 1.0)
        penalized_drift = max(abs_x_drift - self.x_drift_penalty_start, 0.0)
        normalized_penalty = np.clip(penalized_drift / penalty_span, 0.0, 1.0)
        # Make moderate head sway almost free and reserve strong penalties for clearly excessive drift.
        x_drift_penalty = float(normalized_penalty ** 2)
        return signed_x_drift, abs_x_drift, x_drift_norm, x_drift_penalty

    def _update_filtered_motion_state(self, raw_observation):
        alpha = float(np.clip(self.progress_filter_alpha, 0.0, 1.0))
        curr_x_abs = float(raw_observation[0])
        curr_z_abs = float(raw_observation[2])
        curr_heading_norm = float(raw_observation[3])

        if self.filtered_z is None:
            self.filtered_z = curr_z_abs
            self.filtered_x = curr_x_abs
            self.filtered_heading = curr_heading_norm
        else:
            self.filtered_z = alpha * curr_z_abs + (1.0 - alpha) * self.filtered_z
            self.filtered_x = alpha * curr_x_abs + (1.0 - alpha) * self.filtered_x
            self.filtered_heading = alpha * curr_heading_norm + (1.0 - alpha) * self.filtered_heading

        curr_filtered_distance_to_goal = abs(self.targetPositionZ - self.filtered_z)
        if self.prev_filtered_distance_to_goal is None:
            filtered_distance_progress_cm = 0.0
        else:
            filtered_distance_progress_cm = (
                self.prev_filtered_distance_to_goal - curr_filtered_distance_to_goal
            )
        self.prev_filtered_distance_to_goal = curr_filtered_distance_to_goal
        return curr_filtered_distance_to_goal, filtered_distance_progress_cm

    def _compute_signed_progress_reward(self, filtered_distance_progress_cm):
        progress_span = max(self.progress_fullscale_cm - self.progress_deadzone_cm, 1e-6)
        if abs(filtered_distance_progress_cm) <= self.progress_deadzone_cm:
            return 0.0

        magnitude = (
            abs(filtered_distance_progress_cm) - self.progress_deadzone_cm
        ) / progress_span
        signed_progress = np.sign(filtered_distance_progress_cm) * np.clip(magnitude, 0.0, 1.0)
        return float(signed_progress)

    def _compute_window_progress_penalty(self, curr_filtered_distance_to_goal):
        self.filtered_distance_window.append(float(curr_filtered_distance_to_goal))
        if len(self.filtered_distance_window) < (self.progress_window_size + 1):
            return 0.0, 0.0

        window_progress_cm = float(
            self.filtered_distance_window[0] - self.filtered_distance_window[-1]
        )
        progress_deficit_cm = self.window_progress_threshold_cm - window_progress_cm
        if progress_deficit_cm <= 0.0:
            return window_progress_cm, 0.0

        normalized_deficit = np.clip(
            progress_deficit_cm / max(self.window_progress_threshold_cm, 1e-6),
            0.0,
            1.0,
        )
        no_progress_penalty = float(normalized_deficit * self.no_progress_penalty_max)
        return window_progress_cm, no_progress_penalty

    def _build_policy_observation(self, raw_observation):
        raw_observation = np.asarray(raw_observation, dtype=np.float32)

        x_abs = raw_observation[0]
        z_abs = raw_observation[2]
        heading_norm = float(np.clip(raw_observation[3], -1.0, 1.0))
        motor_positions_norm = self._normalize_motor_positions(raw_observation[4:])

        max_distance = max(float(self.targetDistanceZ), 1.0)
        _, _, x_drift_norm, _ = self._compute_x_drift(x_abs)

        if self.starting_position_z is None:
            z_progress_norm = 0.0
            z_remaining_norm = 0.0
        else:
            z_progress = self.progress_direction_z * (z_abs - self.starting_position_z) / max_distance
            z_remaining = self.progress_direction_z * (self.targetPositionZ - z_abs) / max_distance
            z_progress_norm = float(np.clip(z_progress, -1.0, 1.0))
            z_remaining_norm = float(np.clip(z_remaining, -1.0, 1.0))

        observation = np.concatenate([
            np.array([x_drift_norm, z_progress_norm, z_remaining_norm, heading_norm], dtype=np.float32),
            motor_positions_norm.astype(np.float32),
            SnakeEnv.config_numpy.astype(np.float32),
        ])
        return observation

    def step(self, action):

        num_timesteps = 1  
        assert len(action) == 7 * num_timesteps, f"Action space must now be {7 * num_timesteps} values ({7} motors x {num_timesteps} timesteps)."

        # Split action into multiple consecutive timesteps
        actions = [action[i * 7: (i + 1) * 7] for i in range(num_timesteps)]

        for sub_action in actions:
            actionForMotors = self.denormalizeAction(sub_action)
            print(actionForMotors)
            self.writeAction(actionForMotors)

        # Wait briefly and collect a fresh raw observation after the action.
        for i in range(3):
            raw_next_obs = self._get_raw_obs()

        for i in range(5):
            raw_next_obs = self._get_raw_obs()

        tmp_pos = copy.deepcopy(raw_next_obs)

        max_wait = 50
        wait_i = 0
        eps = 1e-3
        delta_x = tmp_pos[0] - self._prev_raw_obs[0]
        delta_z = tmp_pos[2] - self._prev_raw_obs[2]
        while abs(delta_x) < eps and abs(delta_z) < eps and wait_i < max_wait:
            raw_next_obs = self._get_raw_obs()
            tmp_pos = copy.deepcopy(raw_next_obs)
            delta_x = tmp_pos[0] - self._prev_raw_obs[0]
            delta_z = tmp_pos[2] - self._prev_raw_obs[2]
            wait_i += 1

        self._prev_raw_obs = tmp_pos

        # Log global positions
        SnakeEnv.optiXTrack.append(SnakeEnv.optiRelPos[0])  # global x position of robot
        SnakeEnv.optiYTrack.append(SnakeEnv.optiRelPos[2])  # global y position of robot

        # extract Z position from absolute observation (not delta)
        currZPos = tmp_pos[2]  # opti Z position of the robot (absolute, env units)

        # Track both the raw head motion and the short-horizon filtered motion.
        prev_distance_to_goal = abs(self.targetPositionZ - self.prevPos)
        curr_distance_to_goal = abs(self.targetPositionZ - currZPos)
        raw_distance_progress_cm = prev_distance_to_goal - curr_distance_to_goal
        curr_filtered_distance_to_goal, distance_progress_cm = self._update_filtered_motion_state(tmp_pos)
        progress_reward = self._compute_signed_progress_reward(distance_progress_cm)
        window_progress_cm, no_progress_penalty = self._compute_window_progress_penalty(
            curr_filtered_distance_to_goal
        )

        # check if the goal is reached along Z axis
        if self.starting_position_z is not None and self.targetPositionZ < self.starting_position_z:
            terminated = currZPos <= self.targetPositionZ
        else:
            terminated = currZPos >= self.targetPositionZ
        print(f"Step check: currZ={currZPos:.3f}, targetZ={self.targetPositionZ:.3f}, terminated={terminated}")

        # Score net locomotion, not raw head sway.
        _, x_drift, _, x_drift_penalty = self._compute_x_drift(self.filtered_x)
        heading_error = max(abs(self.filtered_heading) - self.heading_penalty_deadzone, 0.0)
        heading_penalty = float(
            np.clip(
                heading_error / max(1.0 - self.heading_penalty_deadzone, 1e-6),
                0.0,
                1.0,
            )
        )
        living_penalty = self.step_living_penalty
        backward_penalty = 0.0

        reward = (
            progress_reward
            - living_penalty
            - no_progress_penalty
            - backward_penalty
            - (self.x_drift_penalty_scale * x_drift_penalty)
            - (self.heading_penalty_scale * heading_penalty)
        )
        if terminated:
            reward += self.terminal_reward_bonus
        reward = float(np.clip(reward, self.reward_clip_min, self.reward_clip_max))
        print(
            f"reward components -> progress_reward: {progress_reward:.4f}, "
            f"raw_distance_progress_cm: {raw_distance_progress_cm:.4f}, "
            f"filtered_distance_progress_cm: {distance_progress_cm:.4f}, "
            f"window_progress_cm: {window_progress_cm:.4f}, "
            f"prev_distance_to_goal: {prev_distance_to_goal:.4f}, curr_distance_to_goal: {curr_distance_to_goal:.4f}, "
            f"curr_filtered_distance_to_goal: {curr_filtered_distance_to_goal:.4f}, "
            f"x_drift_cm: {x_drift:.4f}, x_drift_penalty: {x_drift_penalty:.4f}, "
            f"filtered_heading_penalty: {heading_penalty:.4f}, living_penalty: {living_penalty:.4f}, "
            f"no_progress_penalty: {no_progress_penalty:.4f}"
        )

        truncated = False
        info = {
            'info': 0,
            'progress_reward': progress_reward,
            'progress_step_reward': progress_reward,
            'distance_progress_cm': distance_progress_cm,
            'raw_distance_progress_cm': raw_distance_progress_cm,
            'window_progress_cm': window_progress_cm,
            'step_reward': reward,
            'total_step_reward': reward,
            'x_drift_penalty': x_drift_penalty,
            'heading_penalty': heading_penalty,
            'living_penalty': living_penalty,
            'no_progress_penalty': no_progress_penalty,
            'backward_penalty': backward_penalty,
            'stagnation_penalty': no_progress_penalty,
            'prev_distance_to_goal': prev_distance_to_goal,
            'curr_distance_to_goal': curr_distance_to_goal,
            'curr_filtered_distance_to_goal': curr_filtered_distance_to_goal,
        }

        print(f"Reward: {reward}")

        observation = self._build_policy_observation(tmp_pos)

        # Log data
        self.df.loc[len(self.df.index)] = [
            actionForMotors,
            observation.tolist(),
            list(tmp_pos[4:]),
            reward,
        ]

        # update previous position and action
        self.prevPos = currZPos

        print(f"Observation: {observation}")
        return np.array(observation, dtype=np.float32), reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        # returns: observation of the initial state
        # Keep trajectory logs episode-local so per-episode CSV rows are aligned.
        SnakeEnv.optiXTrack = []
        SnakeEnv.optiYTrack = []
       
        
        super().reset(seed=seed)  # this is needed for cutom environments according to AI Gym
        self.starting_angle = None
        prompt_for_reset = self._should_prompt_for_reset(options)
        auto_motor_reset = self._should_auto_motor_reset(options)
        reset_prompt = 'Reset robot by hand, then press a button to continue'
        if options and options.get('reset_prompt'):
            reset_prompt = str(options['reset_prompt'])

        if prompt_for_reset or not auto_motor_reset:
            self._disable_motor_torque_for_manual_reset()

        if prompt_for_reset:
            try:
                input(reset_prompt)
            except EOFError:
                print('Reset prompt skipped: stdin is not interactive.')
   


        # choose starting position of robot motors
        #startPos = random.sample(range(self.motorMin, self.motorMax), 7)
        if auto_motor_reset:
            startPos = [2048] * len(SnakeEnv.motors.DXL_ID)
            SnakeEnv.motorLock.acquire()
            try:
                reset_ok = SnakeEnv.motors.resetMotorPositions(
                    startPos,
                    disable_after_reset=True,
                )
            finally:
                SnakeEnv.motorLock.release()

            if not reset_ok:
                raise MotorFaultError(
                    "Failed to reset motors to the start position. "
                    "See the hardware error status code output above."
                )

            print('motors reset to start pose')
        else:
            self._refresh_manual_reset_motor_position()
            print('Automatic motor reset disabled; using manual/current motor pose as reset state.')
        # choose new goal position? could randomize target position?
        # self.targetPosition = 
        
        time.sleep(1)   

        # to fill data for reset

        """
        SnakeEnv.optiLock.acquire()
        SnakeEnv.prevPos = SnakeEnv.optiPosition[0:3]
        SnakeEnv.optiLock.release()
        """

        # return current observation
        print("about to observe")
        raw_observation = self._get_raw_obs(initial=True)
        starting_z_abs = raw_observation[2]
        starting_x_abs = raw_observation[0]
        if not SnakeEnv.enableMotorTorque():
            recovered = SnakeEnv.recoverMotorFault(
                context="reset torque enable failed",
                force_reboot=True,
            )
            if not recovered:
                raise MotorFaultError(
                    "Failed to enable motor torque after reset. "
                    "Automatic DYNAMIXEL recovery/reboot was attempted."
                )
        # DO NOT UNCOMMENT IN

        self.starting_position = raw_observation[0]
        self.starting_position_x = starting_x_abs
        self.starting_position_z = starting_z_abs
        self.progress_direction_z = -1.0 if self.targetPositionZ < self.starting_position_z else 1.0
        self.targetDistanceZ = abs(self.targetPositionZ - self.starting_position_z)
        print(
            f"Episode target Z set to {self.targetPositionZ:.3f} (startZ {self.starting_position_z:.3f}, "
            f"distance {self.targetDistanceZ:.3f}, direction {self.progress_direction_z:+.0f})"
        )
        self.prevPos = starting_z_abs
        self.prevXpos = starting_x_abs
        self.filtered_z = starting_z_abs
        self.filtered_x = starting_x_abs
        self.filtered_heading = float(raw_observation[3])
        self.prev_filtered_distance_to_goal = abs(self.targetPositionZ - self.filtered_z)
        self.filtered_distance_window.clear()
        self.filtered_distance_window.append(float(self.prev_filtered_distance_to_goal))
        self._prev_raw_obs = copy.deepcopy(raw_observation)

        observation = self._build_policy_observation(raw_observation)
        print('Observation: ', observation)

        info = {'info': 0}
        return (np.array(observation, dtype=np.float32), info)

    def render(self):
        # graphical window
        # leave empty if not giving user a way to visualize 
        pass

    def close(self):
        # use this to close any files or at the end of sequence
        pass

    def seed(self, seed = None):
        # can use this method to create a random seed
        pass

    def _get_raw_obs(self, initial=False):
        # read agent x,y,z,etc and target goal pos
        self.agentPos = self.getPosition(initial)
        return [*self.agentPos]

    def _get_obs(self, initial=False):
        raw_observation = self._get_raw_obs(initial)
        return self._build_policy_observation(raw_observation)

    def getPosition(self, initial):
       
        # motorPos = self.motors.readPos()
        # optiPos = self.opti.optiTrackGetPos() # currently returning x, y, z
        # self.currPosition = [*optiPos, *motorPos]  
        
 
        SnakeEnv.motorLock.acquire()
        SnakeEnv.optiLock.acquire()


        wait_count = 0
        while len(SnakeEnv.optiPosition) < 6 and wait_count < 1000:
            SnakeEnv.optiLock.release()
            SnakeEnv.motorLock.release()
            time.sleep(.001)
            SnakeEnv.motorLock.acquire()
            SnakeEnv.optiLock.acquire()
            wait_count += 1

        if len(SnakeEnv.optiPosition) < 6:
            SnakeEnv.optiPosition = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # OptiTrack may stop publishing when stationary; keep the last valid
        # pose instead of forcing zeros, but emit a throttled warning.
        if SnakeEnv.opti_last_update_time > 0 and (time.time() - SnakeEnv.opti_last_update_time) > 1.0:
            now = time.time()
            if now - SnakeEnv.opti_last_stale_warn_time > 2.0:
                print('OptiTrack stream stale (>1s). Reusing last valid pose (stationary robot likely).')
                SnakeEnv.opti_last_stale_warn_time = now


        if initial == True:
            SnakeEnv.prevPos = SnakeEnv.optiPosition[0:3]

        #optiPositionCoord = [(curr- prev)*100 for curr, prev in zip(SnakeEnv.optiPosition[0:3], SnakeEnv.prevPos)] # adjusting position to measure previous
        optiPositionCoord_global = [(curr)*100 for curr, _ in zip(SnakeEnv.optiPosition[0:3], SnakeEnv.prevPos)]
        #optiAngle = [i/100 for i in SnakeEnv.optiPosition[3:6]] # only accessing y heading 
        optiAngle = SnakeEnv.optiPosition[5] #/100
        if self.starting_angle is None:
            self.starting_angle = optiAngle

        angle = self.starting_angle - optiAngle
        optiAngle = float((angle + 180.) % 360) - 180.
        optiAngle = optiAngle/180.
        # print('MOTOR POS', SnakeEnv.motorPosition)

        #while SnakeEnv.motorPosition == []:
        #    SnakeEnv.motorLock.release()
        #    time.sleep(.001)
        #    SnakeEnv.motorLock.acquire()
        #print('CHANGE')
        self.currPosition = [*optiPositionCoord_global, optiAngle, *SnakeEnv.motorPosition] # reads static variables that are being updated in the threads 
        
        while SnakeEnv.motorPosition == []:
            SnakeEnv.motorLock.release()
            time.sleep(.001)
            SnakeEnv.motorLock.acquire()
            self.currPosition = [*optiPositionCoord_global, optiAngle, *SnakeEnv.motorPosition]

        SnakeEnv.prevPos = SnakeEnv.optiPosition[0:3]

        SnakeEnv.motorLock.release()
        SnakeEnv.optiLock.release()
        
        time.sleep(.001)
        return self.currPosition
    
    def writeAction(self, actionToWrite):
        posTo = actionToWrite
        print('POSITION TO', posTo)
        SnakeEnv.motorLock.acquire()
        try:
            write_ok = SnakeEnv.motors.writePos(posTo)
        finally:
            SnakeEnv.motorLock.release()

        if not write_ok:
            raise MotorFaultError(
                f"Failed to write motor command {posTo}. "
                "Automatic DYNAMIXEL recovery/reboot was attempted."
            )

        time.sleep(.3) # sleep to allow motors to get to position
        return write_ok


    def getTorque(self):
        motorTor = self.motors.readTorque(self.motorLock)
        return motorTor
    
    def denormalizeAction(self, action):
        
        motorMax = self.motorMax
        motorMin = self.motorMin
        action = np.clip(action, -1.0, 1.0)
        mapping = interp1d([-1, 1], [motorMin, motorMax])
        mappedList = [int(mapping(i)) for i in action]

        return mappedList

    '''
        The following methods are static so they can be accessed from outside environment wrapper to edit parameters with threading 
    '''
    @staticmethod
    def _ensure_hardware_initialized():
        if SnakeEnv.motors is None:
            SnakeEnv.motors = motorssynced.MotorsSynced()
        if SnakeEnv.opti is None:
            SnakeEnv.opti = optitrack.Optitrack()

    @staticmethod
    def passLocksToEnv(oLock, mLock):
        # function to pass locks into this environment
        SnakeEnv.optiLock = oLock
        SnakeEnv.motorLock = mLock
        return
    
    @staticmethod
    def optiPos():
        SnakeEnv._ensure_hardware_initialized()
        SnakeEnv.optiLock.acquire()
        try:
            SnakeEnv.optiRelPos, heading = SnakeEnv.opti.optiTrackGetPos()
            if SnakeEnv.optiRelPos is not None and heading is not None and len(SnakeEnv.optiRelPos) >= 3 and len(heading) >= 3:
                SnakeEnv.optiPosition = [*SnakeEnv.optiRelPos[:3], *heading[:3]]
                now = time.time()
                SnakeEnv.opti_last_update_time = now
                if now - SnakeEnv.opti_last_print_time >= 0.5:
                    x, y, z = SnakeEnv.optiRelPos[:3]
                    #print(f"OptiTrack position (m): x={x:.4f}, y={y:.4f}, z={z:.4f}")
                    SnakeEnv.opti_last_print_time = now
        except Exception as e:
            print(f"Opti thread warning: {e}")
        finally:
            SnakeEnv.optiLock.release()
        time.sleep(.001) # changed from .008
        #SnakeEnv.bla = time.time()
        return
    

    
    @staticmethod
    def motorPos():
        SnakeEnv._ensure_hardware_initialized()
        SnakeEnv.motorLock.acquire()
        try:
            SnakeEnv.motorPosition = SnakeEnv.motors.readPos()
            # print('In thread', SnakeEnv.motorPosition)
        finally:
            SnakeEnv.motorLock.release()
        time.sleep(.001)
        return
    
    @staticmethod
    def returnOptiXList():
        return SnakeEnv.optiXTrack, SnakeEnv.optiYTrack
    

    @staticmethod
    def disableMotorTorque():
        SnakeEnv._ensure_hardware_initialized()
        SnakeEnv.motorLock.acquire()
        try:
            return SnakeEnv.motors.disableTorque()
        finally:
            SnakeEnv.motorLock.release()

    @staticmethod
    def enableMotorTorque():
        SnakeEnv._ensure_hardware_initialized()
        SnakeEnv.motorLock.acquire()
        try:
            return SnakeEnv.motors.enableTorque()
        finally:
            SnakeEnv.motorLock.release()

    @staticmethod
    def recoverMotorFault(context, motor_ids=None, force_reboot=True):
        SnakeEnv._ensure_hardware_initialized()
        SnakeEnv.motorLock.acquire()
        try:
            return SnakeEnv.motors.recoverFromFault(
                context=context,
                motor_ids=motor_ids,
                force_reboot=force_reboot,
            )
        finally:
            SnakeEnv.motorLock.release()

    @staticmethod
    def set_new_design(design):
        coerced = SnakeEnv._coerce_design_vector(design)
        SnakeEnv.current_design = coerced
        SnakeEnv.config_numpy = SnakeEnv.encode_design_vector(coerced)

    @staticmethod
    def _coerce_design_vector(design):
        bounds = SnakeEnv.design_parameter_bounds
        default = list(SnakeEnv.current_design)
        try:
            values = np.asarray(design, dtype=float).reshape(-1).tolist()
        except (TypeError, ValueError):
            values = []
        values = values[:len(bounds)]
        if len(values) < len(bounds):
            values.extend(default[len(values):len(bounds)])

        coerced = []
        for value, (lower, upper) in zip(values, bounds):
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                numeric_value = float(lower)
            if not np.isfinite(numeric_value):
                numeric_value = float(lower)
            coerced.append(float(np.clip(numeric_value, lower, upper)))
        return coerced

    @staticmethod
    def encode_design_vector(design):
        design = SnakeEnv._coerce_design_vector(design)
        bounds = SnakeEnv.design_parameter_bounds
        encoded = []
        for value, (lower, upper) in zip(design, bounds):
            span = max(float(upper) - float(lower), 1e-6)
            encoded.append(float(np.clip(2.0 * ((value - lower) / span) - 1.0, -1.0, 1.0)))

        width_span = max(bounds[0][1] - bounds[0][0], 1e-6)
        attack_angle_span = max(bounds[1][1] - bounds[1][0], 1e-6)
        delta_width_norm = float(np.clip((design[0] - design[2]) / width_span, -1.0, 1.0))
        delta_attack_angle_norm = float(np.clip((design[1] - design[3]) / attack_angle_span, -1.0, 1.0))
        encoded.extend([delta_width_norm, delta_attack_angle_norm])
        return np.asarray(encoded, dtype=np.float32)

    @staticmethod
    def get_design_feature_labels():
        return list(SnakeEnv.design_feature_names)

    @staticmethod
    def get_observation_feature_labels():
        motor_labels = [f'Motor{i + 1}_Norm' for i in range(7)]
        return [
            'X_Drift_Norm',
            'Z_Progress_Norm',
            'Z_Remaining_Norm',
            'Heading_Norm',
            *motor_labels,
            *SnakeEnv.get_design_feature_labels(),
        ]

    @staticmethod
    def set_current_terrain(terrain_name):
        if terrain_name not in SnakeEnv.terrains:
            raise ValueError(f"Unknown terrain '{terrain_name}'. Use one of {SnakeEnv.terrains}")
        SnakeEnv.current_terrain = terrain_name

    @staticmethod
    def get_current_terrain():
        return SnakeEnv.current_terrain

    @staticmethod
    def get_terrain_id(terrain_name):
        if terrain_name not in SnakeEnv.terrain_name_to_id:
            raise ValueError(f"Unknown terrain '{terrain_name}'. Use one of {SnakeEnv.terrains}")
        return SnakeEnv.terrain_name_to_id[terrain_name]
      
    @staticmethod 
    def get_random_design():
        return [
            float(np.random.uniform(lower, upper))
            for lower, upper in SnakeEnv.design_parameter_bounds
        ]
      
    @staticmethod
    def get_current_design():
        return copy.copy(SnakeEnv.current_design)

    @staticmethod
    def get_default_design():
        return SnakeEnv._coerce_design_vector([0.63, 0.0, 0.63, 0.0])

    @staticmethod
    def get_init_design_parameters(design_mode='heterogeneous'):
        design_mode = str(design_mode).strip().lower()
        if design_mode == 'homogeneous':
            return [
                SnakeEnv._coerce_design_vector(design)
                for design in SnakeEnv.homogeneous_init_design_parameters
            ]
        if design_mode == 'heterogeneous':
            return [
                SnakeEnv._coerce_design_vector(design)
                for design in SnakeEnv.heterogeneous_init_design_parameters
            ]
        raise ValueError("SNAKE_SCALE_DESIGN_MODE must be 'homogeneous' or 'heterogeneous'.")

    @staticmethod
    def get_optimization_bounds(design_mode='heterogeneous'):
        design_mode = str(design_mode).strip().lower()
        if design_mode == 'homogeneous':
            return [SnakeEnv.design_parameter_bounds[0], SnakeEnv.design_parameter_bounds[1]]
        if design_mode == 'heterogeneous':
            return list(SnakeEnv.design_parameter_bounds)
        raise ValueError("SNAKE_SCALE_DESIGN_MODE must be 'homogeneous' or 'heterogeneous'.")

    @staticmethod
    def expand_optimization_design(design, design_mode='heterogeneous'):
        design_mode = str(design_mode).strip().lower()
        try:
            values = np.asarray(design, dtype=float).reshape(-1).tolist()
        except (TypeError, ValueError):
            values = []
        if design_mode == 'homogeneous':
            if len(values) < 2:
                default = SnakeEnv.get_default_design()
                values.extend([default[0], default[1]][len(values):])
            return SnakeEnv._coerce_design_vector([values[0], values[1], values[0], values[1]])
        if design_mode == 'heterogeneous':
            return SnakeEnv._coerce_design_vector(values)
        raise ValueError("SNAKE_SCALE_DESIGN_MODE must be 'homogeneous' or 'heterogeneous'.")

    @staticmethod
    def actual_width_from_ratio(width_ratio):
        return float(width_ratio) * float(SnakeEnv.scale_pitch)

    @staticmethod
    def design_summary(design=None):
        design = SnakeEnv._coerce_design_vector(SnakeEnv.current_design if design is None else design)
        return {
            'A_Width_Ratio': float(design[0]),
            'A_Attack_Angle_Deg': float(design[1]),
            'A_Actual_Width': SnakeEnv.actual_width_from_ratio(design[0]),
            'B_Width_Ratio': float(design[2]),
            'B_Attack_Angle_Deg': float(design[3]),
            'B_Actual_Width': SnakeEnv.actual_width_from_ratio(design[2]),
            'Width_Delta': float(design[0] - design[2]),
            'Attack_Angle_Delta': float(design[1] - design[3]),
        }

    @staticmethod
    def expand_design_to_modules(design=None):
        design = SnakeEnv._coerce_design_vector(SnakeEnv.current_design if design is None else design)
        summary = SnakeEnv.design_summary(design)
        modules = []
        for idx, group in enumerate(SnakeEnv.module_group_pattern):
            prefix = 'A' if group == 'A' else 'B'
            modules.append({
                'module': idx + 1,
                'group': group,
                'width_ratio': summary[f'{prefix}_Width_Ratio'],
                'actual_width': summary[f'{prefix}_Actual_Width'],
                'attack_angle_deg': summary[f'{prefix}_Attack_Angle_Deg'],
            })
        return modules

    @staticmethod
    def format_design_for_terminal(design=None):
        summary = SnakeEnv.design_summary(SnakeEnv.current_design if design is None else design)
        return (
            "Scale A: width_ratio={A_Width_Ratio:.3f}, actual_width={A_Actual_Width:.3f}, "
            "attack_angle={A_Attack_Angle_Deg:.2f} deg | "
            "Scale B: width_ratio={B_Width_Ratio:.3f}, actual_width={B_Actual_Width:.3f}, "
            "attack_angle={B_Attack_Angle_Deg:.2f} deg"
        ).format(**summary)
  
    @staticmethod
    def get_design_dimensions():
        return copy.copy(SnakeEnv.design_dims)

    @staticmethod
    def get_number_of_init_designs():
        print('NUM DESIGNS', len(SnakeEnv.get_init_design_parameters()))
        return len(SnakeEnv.get_init_design_parameters())
