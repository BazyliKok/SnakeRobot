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
       Observation Space: 4 normalized motion features + 7 normalized motors + 12 one-hot design features
    '''

    # setting up design framework
    #TODO: make it 1.80 for 1 segment

    # Design vector encodes the scale type used on [head, body, tail].
    # 0 = TPU with spikes, 1 = TPU without spikes,
    # 2 = PLA with spikes, 3 = PLA without spikes.
    current_design = [0, 0, 0]
    scale_types = {
        0: 'TPU_SPIKES',
        1: 'TPU_NO_SPIKES',
        2: 'PLA_SPIKES',
        3: 'PLA_NO_SPIKES',
    }
    terrains = ['carpet', 'cardboard', 'artificial_grass']
    current_terrain = terrains[0]

    # Discrete bounds for [head, body, tail] scale type ids.
    design_parameter_bounds = [(0,3), (0,3), (0,3)]

    
    """
    init_design_parameters = [
            [1, 1, 1, 1, 1, 1],
            [.5, .5, .5, .5, .5, .5],
            [.5, 1, .5, 1, .5, 1],
            [.75, .5, .75, .5, 1, 1]
            ] # NOTE: Change these depending on the design I am going to use
    
    init_design_parameters = [
        [1.80] * 8,
        [.60] * 8,
        [2.70] * 8,
        [1.80, .60, 2.70, 1.80, .60, 2.70, 1.80,.60,],
        [2.653, 1.280, 2.385, 3.191, 1.485, 2.175, .542],

    # ] # NOTE: Change these depending on the design I am going to use
    """
        # Initial symmetric and asymmetric scale-distribution seeds.
    init_design_parameters = [
        [0, 0, 0],  # symmetric: TPU spikes everywhere
        [1, 1, 1],  # symmetric: TPU no spikes
        [2, 2, 2],  # symmetric: PLA spikes
        [3, 3, 3],  # symmetric: PLA no spikes
        [0, 1, 2],  # asymmetric
        [2, 0, 3],  # asymmetric
        [3, 1, 0],  # asymmetric
    ]

    design_slot_names = ['Head', 'Body', 'Tail']
    config_numpy = np.eye(len(scale_types), dtype=np.float32)[current_design].reshape(-1)
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
        self.targetPositionZ = -80.0
        self.x_drift_penalty_start = 15.0
        self.x_drift_penalty_full = 80.0
        self.x_drift_observation_scale = 80.0
        self.progress_filter_alpha = 0.25
        # For a ~1.5 m robot with only head tracking, ignore small recoil/bob
        # and only saturate the progress reward on clearer net displacement.
        self.progress_deadzone_cm = 2.0
        self.progress_fullscale_cm = 6.0
        self.progress_window_size = 6
        self.window_progress_threshold_cm = 3.0
        self.no_progress_penalty_max = 0.15
        self.step_living_penalty = 0.02
        self.heading_penalty_deadzone = 25.0 / 180.0
        self.x_drift_penalty_scale = 0.10
        self.heading_penalty_scale = 0.05
        self.terminal_reward_bonus = 5.0
        self.reward_clip_min = -3.0
        self.reward_clip_max = 5.0
        self._interactive_reset_default = self._read_interactive_reset_default()

               
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
        
     
    def _read_interactive_reset_default(self):
        env_value = os.getenv('SNAKE_INTERACTIVE_RESET')
        if env_value is None:
            return sys.stdin.isatty()
        return env_value.strip().lower() in ('1', 'true', 'yes', 'on')

    def _should_prompt_for_reset(self, options=None):
        if options and 'interactive_reset' in options:
            return bool(options['interactive_reset'])
        return self._interactive_reset_default

    def _normalize_motor_positions(self, motor_positions):
        motor_positions = np.asarray(motor_positions, dtype=np.float32)
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
        if self._should_prompt_for_reset(options):
            try:
                input('Reset robot then press a button to continue')
            except EOFError:
                print('Reset prompt skipped: stdin is not interactive.')
   


        # choose starting position of robot motors
        #startPos = random.sample(range(self.motorMin, self.motorMax), 7)
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
            raise RuntimeError(
                "Failed to reset motors to the start position. "
                "See the hardware error status code output above."
            )

        print('motors reset to start pose')
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
        SnakeEnv.enableMotorTorque()
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
            SnakeEnv.motors.disableTorque()
        finally:
            SnakeEnv.motorLock.release()
        #time.sleep(.005)
        return   

    @staticmethod
    def enableMotorTorque():
        SnakeEnv._ensure_hardware_initialized()
        SnakeEnv.motorLock.acquire()
        try:
            SnakeEnv.motors.enableTorque()
        finally:
            SnakeEnv.motorLock.release()
        #time.sleep(.005)
        return   

    @staticmethod
    def set_new_design(design):
        # Keep design ids in [0..3] and integer-coded.
        rounded = [int(np.clip(np.round(v), 0, 3)) for v in design]
        SnakeEnv.current_design = rounded
        SnakeEnv.config_numpy = SnakeEnv.encode_design_vector(rounded)

    @staticmethod
    def encode_design_vector(design):
        design_ids = [int(np.clip(np.round(v), 0, len(SnakeEnv.scale_types) - 1)) for v in design]
        encoded = np.zeros(len(design_ids) * len(SnakeEnv.scale_types), dtype=np.float32)
        for idx, design_id in enumerate(design_ids):
            encoded[idx * len(SnakeEnv.scale_types) + design_id] = 1.0
        return encoded

    @staticmethod
    def get_design_feature_labels():
        labels = []
        scale_ids = sorted(SnakeEnv.scale_types.keys())
        for slot_idx in range(len(SnakeEnv.current_design)):
            if slot_idx < len(SnakeEnv.design_slot_names):
                slot_name = SnakeEnv.design_slot_names[slot_idx]
            else:
                slot_name = f'Segment{slot_idx + 1}'
            for scale_id in scale_ids:
                labels.append(f'{slot_name}_{SnakeEnv.scale_types[scale_id]}')
        return labels

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
    def get_random_design():
        optimized_params = np.random.uniform(
            low=SnakeEnv.design_parameter_bounds[0][0],
            high=SnakeEnv.design_parameter_bounds[0][1],
            size=len(SnakeEnv.design_parameter_bounds),
        )
        return optimized_params
      
    @staticmethod
    def get_current_design():
        return copy.copy(SnakeEnv.current_design)
  
    @staticmethod
    def get_design_dimensions():
        return copy.copy(SnakeEnv.design_dims)

    @staticmethod
    def get_number_of_init_designs():
        print('NUM DESIGNS', len(SnakeEnv.init_design_parameters))
        return len(SnakeEnv.init_design_parameters)
