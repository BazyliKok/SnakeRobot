import json
import gymnasium as gym
import matplotlib.pyplot as plt
from soft_actor_critic_coadapt import SoftActorCriticCoadapt
from snakeenv_thread_coadapt import SnakeEnv
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
from motorssynced import MotorsSynced
from collections import Counter

from datetime import datetime

class Train():
    def __init__(self):        

        self.env = gym.make("SnakeRobot")
    
        self._reward_scale = 1.0
        self.optimized_params = None
        self._episode_length = 250 # number of timesteps per episode
        self.episode_counter = None
        self.policy_action_warmup_episodes = 0  # full-random episodes before policy/random mixing starts
        self.training_update_warmup_episodes = 1  # collected episodes before SAC updates start
        self.design_cylces = 20 # total number of design cycles
        self.terrain_sequence = list(SnakeEnv.terrains)
        self.training_terrain_block_size = 8
        self.episode_iterations = len(self.terrain_sequence) * self.training_terrain_block_size # number of episodes per design
        self.results_tag = 'mixed_terrain'
        self.legacy_results_tags = [self.results_tag, 'carpet', 'carton', 'foam']
        self.terrain_name_to_id = {terrain: idx for idx, terrain in enumerate(self.terrain_sequence)}
        self.training_terrain_block_order = []
        self.training_episode_schedule = []
        self.training_schedule_design_counter = None
        self.current_training_terrain = None
        self.current_training_terrain_id = -1
        self.current_training_block_index = -1
        self.current_training_episode_in_block = -1
        self.current_schedule_seed = None
        self._last_seed_context = None
        self.current_episode_seed = None
        self.current_update_seed = None
        self.current_design_optimization_seed = None

        # Keep commands changing so the robot does not lock into one saturated pose.
        self.action_noise_std = 0.10
        self.repeat_action_eps = 0.02
        self.repeat_action_perturb_std = 0.08
        self.random_action_prob_start = 0.3
        self.random_action_prob_decay = 0.02
        self.random_action_prob_min = 0.1

        self.episodeCumulativeRewards = []  # Stores cumulative rewards per episode
        self.cumulativeRewards = []  # Stores cumulative rewards per step

        self.episodeCumulativeRewards = []

        self.eachEpisodeCumuRewards = []

        self.num_init_designs = 7 # number of initial design cycles
        self.seed = int(os.getenv('SNAKE_EXPERIMENT_SEED', '12345'))
        self.eval_episodes_per_terrain = max(1, int(os.getenv('SNAKE_EVAL_EPISODES_PER_TERRAIN', '5')))
        self.eval_robustness_lambda = 0.5
        # set up replay
        self.replay = CoadaptReplayBuffer(
            max_replay_buffer_size_species=int(1e6),
            max_replay_buffer_size_population=int(1e7),
            env= self.env,
            env_info_sizes=None
        )
        self._seed_global_rngs('initialization')

        # set up RL algorithm
        self.rl_method = SoftActorCriticCoadapt
        self.networks = self.rl_method.create_networks(env=self.env)
        self.rl_alg = self.rl_method(env=self.env, replay=self.replay, networks=self.networks)

        # set up design variables
        self.do_alg = PSO_batch(self.replay, self.env)
        self.design_counter = 0
        self.data_design_type = 'Initial'
        

        self.date = datetime.now().strftime("%Y_%m_%d") # for files
        
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
        return episode_idx >= self.training_update_warmup_episodes

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
        self.actionList = [[] for _ in range(self._action_dim())] #was 6
        self.designList = [[] for i in range(0,7)]
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
        self.progressRewardComponents = []
        self.distanceProgressCmComponents = []
        self.xDriftPenaltyComponents = []
        self.headingPenaltyComponents = []
        self.noProgressPenaltyComponents = []
        self.backwardPenaltyComponents = []

    def _set_output_filenames(self):
        self.date = datetime.now().strftime("%Y_%m_%d")
        name = "Rewards_Design{}_{}".format(str(self.design_counter), self.results_tag)
        self.filename = self.date+name
        name = "Losses_Design{}_{}".format(str(self.design_counter), self.results_tag)
        self.lossFilename = self.date+name

    def _checkpoint_results_dir(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_bazyli')


    def _build_randomized_training_schedule(self):
        self.current_schedule_seed = self._stable_seed('terrain_schedule', self.design_counter)
        schedule_rng = np.random.default_rng(self.current_schedule_seed)
        terrain_block_order = list(schedule_rng.permutation(self.terrain_sequence))
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
        if Counter(self.training_terrain_block_order) != Counter(self.terrain_sequence):
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
                'No saved mixed-terrain schedule found for this design; '
                'generating a new randomized terrain block order for the remaining episodes.'
            )

        self.training_terrain_block_order, self.training_episode_schedule = self._build_randomized_training_schedule()
        self.training_schedule_design_counter = self.design_counter
        print(
            f"Design {self.design_counter} terrain block order: "
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

        return tagged_candidates[0]

    def run(self, stopEvent, max_design_cycles_per_run=1):
        """ Runs Fast Evolution through Actor-Critic RL algorithm.
        Chunked execution: process up to max_design_cycles_per_run design cycles
        in one invocation, then return so hardware and metrics can be checked.
        """
        self._initialize_run_logs()
        ptu.set_gpu_mode(False)

        completed_cycles = 0
        while self.design_counter < self.design_cylces and completed_cycles < max_design_cycles_per_run:
            self._set_output_filenames()

            if self.design_counter < self.num_init_designs:
                self.initial_design_loop()
                print(f'design counter at {self.design_counter}')
                if self.design_counter == self.num_init_designs and self.optimized_params is None:
                    self.first_train_op()
            else:
                self.train_loop()

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
            self.actionList = [[] for _ in range(self._action_dim())] # was 6
            self.timestepRewards = []
            self.cumulativeRewards = []
            self.epList = []
            self.timesteps = []
            self.progressRewardComponents = []
            self.distanceProgressCmComponents = []
            self.xDriftPenaltyComponents = []
            self.headingPenaltyComponents = []
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
            state, info = self.env.reset(seed=self.current_episode_seed)
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
                
        
                next_state, reward, terminated, truncated, info = self.env.step(action) # step the action, note: reward is scaled in environment


                
                episodeRewards += reward # accumulate rewards here to track for comparison
        
                # log rewards
                self.timestepRewards.append(reward)
                self.cumulativeRewards.append(episodeRewards)
                self.epList.append(self.currEp) # to make note of what episode we are on
                self.progressRewardComponents.append(float(info.get('progress_reward', np.nan)))
                self.distanceProgressCmComponents.append(float(info.get('distance_progress_cm', np.nan)))
                self.xDriftPenaltyComponents.append(float(info.get('x_drift_penalty', np.nan)))
                self.headingPenaltyComponents.append(float(info.get('heading_penalty', np.nan)))
                self.noProgressPenaltyComponents.append(float(info.get('no_progress_penalty', info.get('stagnation_penalty', np.nan))))
                self.backwardPenaltyComponents.append(float(info.get('backward_penalty', np.nan)))
                for i in range(len(state)): #was 17
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

            SnakeEnv.disableMotorTorque() # stop motors at the end of each episode
            print('disabled torque')


                
             
            self.episodeCumulativeRewards.append(episodeRewards)
            self.eachEpisodeCumuRewards.append(episodeContRewards) # list of a list

            self.logData() # log data
            self.replay.terminate_episode() # run replay end sequence




    def initialize_episode(self):
        """ Initializations required before the first episode.

        Should be called before the first episode of a new design is
        executed. Resets variables such as _data_rewards for logging purposes
        etc.

        """
        #self._rl_alg.initialize_episode(init_networks = True, copy_from_gobal = True)
        self.rl_alg.episode_init()    

        if self.episode_counter == 0:
            self.replay.reset_individual_buffer()


        self.data_rewards = []
    
    def first_train_op(self):
        print('in first train op')
        iterations = self.episode_iterations 
        self.data_design_type = 'Optimized'

        # set up rewards file
        
        #self.episodeFilename = "RewardsEachEpisode_Design{}".format(str(self.design_counter))
        #self.episodeFilename = self.episodeFilename+self.date

        self.initialize_episode()
        
        print(f'design counter at {self.design_counter}')
        if self.design_counter == self.num_init_designs: # change this to mathc num init designs #SnakeEnv.get_number_of_init_designs: # if first time after init design loop
         
            self.current_design_optimization_seed = self._seed_global_rngs(
                'design_opt_bootstrap',
                self.design_counter,
                self.episode_counter,
            )
            self.env.reset(seed=self.current_design_optimization_seed)
            
            self.optimized_params = [0, 0, 0]
            # or can: self.optimized_params = SnakeEnv.get_random_design()
          

            q_network = self.rl_alg.get_q_network(self.networks['population'])
            policy_network = self.rl_alg.get_policy_network(self.networks['population'])
            self.cost, self.optimized_params = self.do_alg.optimize_design(design=self.optimized_params, q_network=q_network, policy_network=policy_network)
            self.optimized_params = list(self.optimized_params)
            print('OPTIMIZED PARAM NEW DESIGN: ', self.optimized_params)
            print('COST: ', self.cost)
        



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
        self.initialize_episode()
        SnakeEnv.set_new_design(self.optimized_params)
        self._ensure_design_training_schedule()

        # Reinforcement Learning
        start_ep = self.episode_counter
        for episode in range(start_ep, iterations):
            print('IN TRAINING LOOP')
            self.currEp = episode
            self.train_single_iteration()
        
            #self.plot_rewards()

        # Evaluate current design before running design optimization
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
            self.cost, self.optimized_params = self.do_alg.optimize_design(design=self.optimized_params, q_network=q_network, policy_network=policy_network)
            self.optimized_params = list(self.optimized_params)
            print('NEW DESIGN PARAMETERS: ',self.optimized_params)
            print('COST: ', self.cost)
        #else: # randomize next design
        #    self._data_design_type = 'Random'
        #    self.optimized_params = SnakeEnv.get_random_design()
        #    self.optimized_params = list(self.optimized_params)

        
        self.design_counter += 1 # another design
        self.episode_counter = 0

        
            
    def train_single_iteration(self):
        
        self.replay.set_mode("species")
        self.collect_training_experience() # collect data
        
        if self.design_counter >= 3: # only train population afer certain number of designs, in this case 3
            train_pop = True
        else:
            train_pop = False
        
        print('train single iteration check if training warmup is complete')
        if self._should_train_updates():  # can start training after enough full episodes are collected
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
            self.epListLoss.extend([self.episode_counter] * len(q1loss))
        self.logTrainLoss() # log data
        self.episode_counter += 1

        print(f'episode counter at: {self.episode_counter}')

        self.save_networks()
      

    def initial_design_loop(self):
        """ The initial training loop for initial designs.

        The initial training loop in which no designs are optimized but only
        initial designs, provided by the environment, are used.

        Args:
            iterations: Integer stating how many training iterations/episodes
                to use per design.

        """
        self.data_design_type = 'Initial'
        params = SnakeEnv.init_design_parameters[self.design_counter] # choose design based on in which design cycle we are

        SnakeEnv.set_new_design(params)
        self.initialize_episode() 
        self._ensure_design_training_schedule()

        
        #for _ in range(self.episode_counter, self.episode_iterations): # train motor controls for this design iteration #added self.episode_counter
        for _ in range(self.episode_counter, self.episode_iterations):
            self.currEp = _
            print('in initial design loop')
            self.train_single_iteration()

            print(f'range {range(self.episode_counter, self.episode_iterations)}')
        
        self.evaluate_policy()
        self.design_counter += 1
        self.episode_counter = 0

        
        return
          
    def evaluate_policy(self):
        """Evaluate deterministic policy performance across all terrains.
        Runs repeated deterministic rollouts per terrain and records richer
        return, success, and rollout-length statistics for each design.
        """
        policy = self.rl_alg.get_policy_network(self.networks['individual'])
        previous_terrain = SnakeEnv.get_current_terrain()

        terrain_returns = {}
        terrain_lengths = {}
        terrain_successes = {}
        terrain_success_steps = {}
        terrain_rollout_seeds = {}

        for terrain_idx, terrain in enumerate(self.terrain_sequence):
            SnakeEnv.set_current_terrain(terrain)
            episode_returns = []
            episode_lengths = []
            episode_successes = []
            success_steps = []
            rollout_seeds = []

            for rollout_idx in range(self.eval_episodes_per_terrain):
                eval_seed = self._stable_seed(
                    'eval_rollout',
                    self.design_counter,
                    terrain_idx,
                    rollout_idx,
                )
                rollout_seeds.append(int(eval_seed))
                state, _ = self.env.reset(seed=eval_seed)
                done = False
                steps = 0
                cumulative_reward = 0.0
                success = False

                while (not done) and steps < self._episode_length:
                    try:
                        action, _ = policy.get_action(state, deterministic=True)
                    except TypeError:
                        action, _ = policy.get_action(state)

                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    cumulative_reward += float(reward)
                    steps += 1
                    success = success or bool(terminated)
                    done = terminated or truncated or (steps >= self._episode_length)
                    state = next_state

                SnakeEnv.disableMotorTorque()
                episode_returns.append(float(cumulative_reward))
                episode_lengths.append(int(steps))
                episode_successes.append(int(success))
                if success:
                    success_steps.append(int(steps))

            terrain_returns[terrain] = episode_returns
            terrain_lengths[terrain] = episode_lengths
            terrain_successes[terrain] = episode_successes
            terrain_success_steps[terrain] = success_steps
            terrain_rollout_seeds[terrain] = rollout_seeds

        SnakeEnv.set_current_terrain(previous_terrain)

        terrain_means = {terrain: float(np.mean(vals)) for terrain, vals in terrain_returns.items()}
        terrain_std = {terrain: float(np.std(vals)) for terrain, vals in terrain_returns.items()}
        terrain_medians = {terrain: float(np.median(vals)) for terrain, vals in terrain_returns.items()}
        terrain_mins = {terrain: float(np.min(vals)) for terrain, vals in terrain_returns.items()}
        terrain_success_rates = {
            terrain: float(np.mean(vals)) if vals else 0.0
            for terrain, vals in terrain_successes.items()
        }
        terrain_mean_lengths = {
            terrain: float(np.mean(vals)) if vals else 0.0
            for terrain, vals in terrain_lengths.items()
        }
        terrain_mean_success_steps = {
            terrain: (float(np.mean(vals)) if vals else None)
            for terrain, vals in terrain_success_steps.items()
        }

        mean_return_per_terrain = np.array(list(terrain_means.values()), dtype=np.float32)
        all_eval_returns = np.array(
            [ret for returns in terrain_returns.values() for ret in returns],
            dtype=np.float32,
        )
        all_eval_lengths = np.array(
            [length for lengths in terrain_lengths.values() for length in lengths],
            dtype=np.float32,
        )
        all_success_steps = np.array(
            [step for steps in terrain_success_steps.values() for step in steps],
            dtype=np.float32,
        )
        mean_return = float(np.mean(mean_return_per_terrain))
        worst_terrain_return = float(np.min(mean_return_per_terrain))
        std_across_terrains = float(np.std(mean_return_per_terrain))
        robustness_score = float(mean_return - self.eval_robustness_lambda * std_across_terrains)
        mean_success_rate = float(np.mean(list(terrain_success_rates.values())))
        worst_success_rate = float(np.min(list(terrain_success_rates.values())))
        overall_median_eval_return = float(np.median(all_eval_returns))
        overall_min_eval_return = float(np.min(all_eval_returns))
        overall_mean_episode_length = float(np.mean(all_eval_lengths))
        overall_mean_success_steps = (
            float(np.mean(all_success_steps)) if len(all_success_steps) > 0 else None
        )

        summary_row = {
            'Date': self.date,
            'Experiment_Seed': int(self.seed),
            'Training_Schedule_Seed': int(self.current_schedule_seed) if self.current_schedule_seed is not None else None,
            'Design_Counter': int(self.design_counter),
            'Episode_Counter': int(self.episode_counter),
            'Scale_Head': int(SnakeEnv.get_current_design()[0]),
            'Scale_Body': int(SnakeEnv.get_current_design()[1]),
            'Scale_Tail': int(SnakeEnv.get_current_design()[2]),
            'Training_Episodes_Per_Design': int(self.episode_iterations),
            'Training_Terrain_Block_Size': int(self.training_terrain_block_size),
            'Training_Terrain_Block_Order': '|'.join(self.training_terrain_block_order),
            'Eval_Episodes_Per_Terrain': int(self.eval_episodes_per_terrain),
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

        for terrain in self.terrain_sequence:
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
            'summary': summary_row,
            'terrain_episode_returns': terrain_returns,
            'terrain_episode_lengths': terrain_lengths,
            'terrain_episode_successes': terrain_successes,
            'terrain_success_steps': terrain_success_steps,
            'terrain_rollout_seeds': terrain_rollout_seeds,
        }
        detail_json_path = os.path.join(
            results_dir,
            f'{self.date}_Design{self.design_counter}_ep{self.episode_counter}_eval_summary.json'
        )
        with open(detail_json_path, 'w') as f:
            json.dump(detail_payload, f, indent=2, allow_nan=False)

        print('Evaluation summary:', summary_row)
       
    def save_networks(self):
        """ Saves the networks on the disk.
        """
         # TODO: Edit this to store more efficiently

        results_dir = self._checkpoint_results_dir()
        os.makedirs(results_dir, exist_ok=True)
        checkpoint_prefix = f'{self.date}_Design{self.design_counter}_ep{self.episode_counter}'

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

        
        metadata = {
            'seed': int(self.seed),
            'design_counter': self.design_counter,
            'episode_counter': self.episode_counter,
            'optimized_params': getattr(self, 'optimized_params', None),
            'training_terrain_block_order': list(self.training_terrain_block_order),
            'training_episode_schedule': list(self.training_episode_schedule),
            'training_schedule_design_counter': self.training_schedule_design_counter,
            'training_terrain_block_size': self.training_terrain_block_size,
            'training_schedule_seed': self.current_schedule_seed,
            'policy_action_warmup_episodes': self.policy_action_warmup_episodes,
            'training_update_warmup_episodes': self.training_update_warmup_episodes,
            'random_action_prob_start': self.random_action_prob_start,
            'random_action_prob_decay': self.random_action_prob_decay,
            'random_action_prob_min': self.random_action_prob_min,
        }

        with open(os.path.join(results_dir, f'{checkpoint_prefix}_metadata_{self.results_tag}.json'), 'w') as f:
            json.dump(metadata, f)

        self.save_replay(os.path.join(results_dir, f'replay_{self.date}_Design{self.design_counter}_{self.results_tag}.pt'))

        print(f"saved networks for design {self.design_counter} and episode {self.episode_counter}")    

    def load_networks(self, base_path, checkpoint_prefix):
        self.rl_alg._ind_policy.load_state_dict(torch.load(
            self._resolve_tagged_path(base_path, f'ind_policy_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._ind_qf1.load_state_dict(torch.load(
            self._resolve_tagged_path(base_path, f'ind_qf1_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._ind_qf2.load_state_dict(torch.load(
            self._resolve_tagged_path(base_path, f'ind_qf2_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._ind_qf1_target.load_state_dict(torch.load(
            self._resolve_tagged_path(base_path, f'ind_qf1_tar_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._ind_qf2_target.load_state_dict(torch.load(
            self._resolve_tagged_path(base_path, f'ind_qf2_tar_{checkpoint_prefix}', 'pt')
        ).state_dict())

        self.rl_alg._pop_policy.load_state_dict(torch.load(
            self._resolve_tagged_path(base_path, f'pop_policy_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._pop_qf1.load_state_dict(torch.load(
            self._resolve_tagged_path(base_path, f'pop_qf1_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._pop_qf2.load_state_dict(torch.load(
            self._resolve_tagged_path(base_path, f'pop_qf2_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._pop_qf1_target.load_state_dict(torch.load(
            self._resolve_tagged_path(base_path, f'pop_qf1_tar_{checkpoint_prefix}', 'pt')
        ).state_dict())
        self.rl_alg._pop_qf2_target.load_state_dict(torch.load(
            self._resolve_tagged_path(base_path, f'pop_qf2_tar_{checkpoint_prefix}', 'pt')
        ).state_dict())

        print(f"loaded networks from checkpoint: {checkpoint_prefix}")

        metadata_path = self._resolve_tagged_path(base_path, f'{checkpoint_prefix}_metadata', 'json')

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.seed = int(metadata.get('seed', self.seed))
            self.design_counter = metadata['design_counter']
            self.episode_counter = metadata['episode_counter']
            self.optimized_params = metadata.get('optimized_params', None)
            self.training_terrain_block_order = metadata.get('training_terrain_block_order', [])
            self.training_episode_schedule = metadata.get('training_episode_schedule', [])
            self.training_schedule_design_counter = metadata.get(
                'training_schedule_design_counter',
                self.design_counter if self.training_episode_schedule else None,
            )
            self.training_terrain_block_size = metadata.get(
                'training_terrain_block_size',
                self.training_terrain_block_size,
            )
            self.current_schedule_seed = metadata.get('training_schedule_seed', self.current_schedule_seed)
            self.policy_action_warmup_episodes = metadata.get(
                'policy_action_warmup_episodes',
                self.policy_action_warmup_episodes,
            )
            self.training_update_warmup_episodes = metadata.get(
                'training_update_warmup_episodes',
                self.training_update_warmup_episodes,
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
            self.episode_iterations = len(self.terrain_sequence) * self.training_terrain_block_size
            self._seed_global_rngs('resume', self.design_counter, self.episode_counter)
            print(f"restored design_counter={self.design_counter}, episode_counter={self.episode_counter}")
        else:
            print("no metadata file found; counters not restored.")

        replay_path = self._resolve_tagged_path(base_path, f'replay_{checkpoint_prefix.split("_ep")[0]}', 'pt')
        if os.path.exists(replay_path):
            self.load_replay(replay_path)
            print("Replay contains", self.replay._individual_buffer._size, "steps")
        else:
            print("no replay buffer found.")



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
        buffer._observation_dim = data.get("_observation_dim", buffer._observation_dim)
        buffer._action_dim = data.get("_action_dim", buffer._action_dim)

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
            data = torch.load(filepath)

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
                return

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
        except Exception as e:
            print(f"failed to load replay buffer: {e}")

    def logData(self):
        os.makedirs(os.path.dirname(self.filename) or '.', exist_ok=True)
        xPositionList, yPositionList = SnakeEnv.returnOptiXList()
        min_len = min(len(self.timesteps), len(xPositionList))

        # trim all lists to the same length
        self.timesteps = self.timesteps[:min_len]
        self.timestepRewards = self.timestepRewards[:min_len]
        self.cumulativeRewards = self.cumulativeRewards[:min_len]
        self.progressRewardComponents = self.progressRewardComponents[:min_len]
        self.distanceProgressCmComponents = self.distanceProgressCmComponents[:min_len]
        self.xDriftPenaltyComponents = self.xDriftPenaltyComponents[:min_len]
        self.headingPenaltyComponents = self.headingPenaltyComponents[:min_len]
        self.noProgressPenaltyComponents = self.noProgressPenaltyComponents[:min_len]
        self.backwardPenaltyComponents = self.backwardPenaltyComponents[:min_len]
        xPositionList = xPositionList[-min_len:]
        yPositionList = yPositionList[-min_len:]
        self.epList = self.epList[:min_len]

        for i in range(len(self.actionList)): #was 6
            self.actionList[i] = self.actionList[i][:min_len]
        for i in range(len(self.stateList)):
            self.stateList[i] = self.stateList[i][:min_len]
        rewardDF = pd.DataFrame()

        rewardDF['Episode'] = [self.episode_counter]* len(self.timesteps)
        rewardDF['Timestep'] = self.timesteps
        rewardDF['X_Position']= xPositionList # added this, need to see if it works
        rewardDF['Y_Position']= yPositionList # added this, need to see if it works
        rewardDF['Rewards'] = self.timestepRewards
        rewardDF['Cumulative_Rewards'] = self.cumulativeRewards
        rewardDF['Progress_Reward'] = self.progressRewardComponents
        rewardDF['Distance_Progress_Cm'] = self.distanceProgressCmComponents
        rewardDF['X_Drift_Penalty'] = self.xDriftPenaltyComponents
        rewardDF['Heading_Penalty'] = self.headingPenaltyComponents
        rewardDF['No_Progress_Penalty'] = self.noProgressPenaltyComponents
        rewardDF['Backward_Penalty'] = self.backwardPenaltyComponents
        rewardDF['Terrain'] = [SnakeEnv.get_current_terrain()] * len(self.timesteps)
        design = SnakeEnv.get_current_design()
        rewardDF['Experiment_Seed'] = [int(self.seed)] * len(self.timesteps)
        rewardDF['Episode_Seed'] = [self.current_episode_seed] * len(self.timesteps)
        rewardDF['Terrain_ID'] = [self.current_training_terrain_id] * len(self.timesteps)
        rewardDF['Terrain_Block_Index'] = [self.current_training_block_index] * len(self.timesteps)
        rewardDF['Episode_In_Terrain_Block'] = [self.current_training_episode_in_block] * len(self.timesteps)
        rewardDF['Scale_Head'] = [int(design[0])] * len(self.timesteps)
        rewardDF['Scale_Body'] = [int(design[1])] * len(self.timesteps)
        rewardDF['Scale_Tail'] = [int(design[2])] * len(self.timesteps)

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

        current_episode = self.episode_counter
        # read existing file if it exists and is valid
        if os.path.isfile(self.filename):
            try:
                existing = pd.read_csv(self.filename)
                # remove old entries of current episode
                existing = existing[existing['Episode'] != current_episode]
                updated = pd.concat([existing, rewardDF], ignore_index=True)
                updated.to_csv(self.filename, index=False)
            except pd.errors.EmptyDataError:
                print(f"{self.filename} is empty. creating new.")
                rewardDF.to_csv(self.filename, index=False)
        else:
            rewardDF.to_csv(self.filename, index=False)

    def logTrainLoss(self):
        os.makedirs(os.path.dirname(self.lossFilename) or '.', exist_ok=True)
        lossDF = pd.DataFrame()
        lossDF['Episode'] = self.epListLoss
        lossDF['Ind_Q1_Loss'] = self.q1loss
        lossDF['Ind_Q2_Loss'] = self.q2loss
        lossDF['Ind_Policy_Loss'] = self.policyloss

         
        lossDF['Pop_Q1_Loss'] = self.popq1loss
        lossDF['Pop_Q2_Loss'] = self.popq2loss
        lossDF['Pop_Policy_Loss'] = self.poppolicyloss
        lossDF.to_csv(self.lossFilename, index=False)


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
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results_bazyli')
    #change name
    checkpoint_prefix = "2025_06_03_Design0_ep30"

    #set to false if new training starts
    resuming_from_checkpoint = False 

    if resuming_from_checkpoint:
        trainingObj.load_networks(base_path, checkpoint_prefix)
    else:
        trainingObj.episode_counter = 0
        print("Starting fresh: episode_counter set to 0")

    # Chunked execution: run a small number of design cycles per launch.
    designs_per_run = 1

    # run threads as before
    motorThread = threading.Thread(target=trainingObj.motorPos, args=(stopEvent,)) 
    optiThread = threading.Thread(target=trainingObj.optiPos, args=(stopEvent,))
    trainingloopThread = threading.Thread(target=trainingObj.run, args=(stopEvent, designs_per_run))
    
    motorThread.start()
    optiThread.start() 
    trainingloopThread.start()
    trainingloopThread.join()

