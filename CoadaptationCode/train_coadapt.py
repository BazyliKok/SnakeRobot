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
from pso_batch import PSO_batch
import time
import rlkit.torch.pytorch_util as ptu
from motorssynced import MotorsSynced

from datetime import datetime

class Train():
    def __init__(self):        

        self.env = gym.make("SnakeRobot")
    
        self._reward_scale = 1.0
        self.optimized_params = None
        self._episode_length = 150 # number of timesteps per episode
        self.episode_counter = None
        self.episodes_before_training = 0 #4 # number of episodes before training to fill the replay buffer
        self.episode_iterations = 32 # number of episodes per design
        self.design_cylces = 20 # total number of design cycles

        # Keep commands changing so the robot does not lock into one saturated pose.
        self.action_noise_std = 0.10
        self.repeat_action_eps = 0.02
        self.repeat_action_perturb_std = 0.08

        self.episodeCumulativeRewards = []  # Stores cumulative rewards per episode
        self.cumulativeRewards = []  # Stores cumulative rewards per step

        self.episodeCumulativeRewards = []

        self.eachEpisodeCumuRewards = []

        self.num_init_designs = 7 # number of initial design cycles
        # set up replay
        self.replay = CoadaptReplayBuffer(
            max_replay_buffer_size_species=int(1e6),
            max_replay_buffer_size_population=int(1e7),
            env= self.env,
            env_info_sizes=None
        )

        # set up RL algorithm
        self.rl_method = SoftActorCriticCoadapt
        self.networks = self.rl_method.create_networks(env=self.env)
        self.rl_alg = self.rl_method(env=self.env, replay=self.replay, networks=self.networks)

        # set up design variables
        self.do_alg = PSO_batch(self.replay, self.env)
        self.design_counter = 0
        self.data_design_type = 'Initial'
        

        self.date = datetime.now().strftime("%Y_%m_%d") # for files
        self.terrain_sequence = ['floor', 'carpet', 'cardboard', 'artificial_grass']
        self.eval_episodes_per_terrain = 3
        self.eval_robustness_lambda = 0.5
        
    def _action_dim(self):
        action_shape = getattr(self.env.action_space, "shape", None)
        if action_shape and len(action_shape) > 0:
            return action_shape[0]
        return 7

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

    def _set_output_filenames(self):
        self.date = datetime.now().strftime("%Y_%m_%d")
        name = "Rewards_Design{}_carpet".format(str(self.design_counter))
        self.filename = self.date+name
        name = "Losses_Design{}_carpet".format(str(self.design_counter))
        self.lossFilename = self.date+name

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

            # reset environment
            terrain_idx = self.episode_counter % len(self.terrain_sequence)
            terrain = self.terrain_sequence[terrain_idx]
            SnakeEnv.set_current_terrain(terrain)
            print(f"CURRENT TERRAIN: {terrain}. Place robot on this terrain before continuing reset.")
            state, info = self.env.reset()
            state_dim = len(state)
            self.stateList = [[] for _ in range(state_dim)]
            steps = 0
            episodeRewards = 0
            episodeContRewards = []
            Done = False
            prev_action = None
        
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
                if self.currEp >= self.episodes_before_training:  # can start training, exploitation
                    action, _ = self.policy.get_action(state)
                else:  # purely exploring
                    action = np.random.uniform(-1, 1, size=7)

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
         
            self.env.reset()
            
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
        
        print('train single iteration check if counter > episodes before training')
        if self.episode_counter > self.episodes_before_training: # can start training, have filled buffer
            print('counter > episodes')
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
        Runs multiple deterministic rollouts per terrain and stores a
        per-design summary for robust cross-terrain comparison.
        """
        # can add a policy rollout here
        policy = self.rl_alg.get_policy_network(self.networks['individual'])

        terrain_returns = {}
        for terrain_idx, terrain in enumerate(self.terrain_sequence):
            SnakeEnv.set_current_terrain(terrain)
            episode_returns = []

            for _ in range(self.eval_episodes_per_terrain):
                state, _ = self.env.reset()
                done = False
                steps = 0
                cumulative_reward = 0.0

                while (not done) and steps < self._episode_length:
                    try:
                        action, _ = policy.get_action(state, deterministic=True)
                    except TypeError:
                        action, _ = policy.get_action(state)

                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    cumulative_reward += float(reward)
                    steps += 1
                    done = terminated or truncated or (steps >= self._episode_length)
                    state = next_state

                SnakeEnv.disableMotorTorque()
                episode_returns.append(cumulative_reward)

            terrain_returns[terrain] = episode_returns

        terrain_means = {terrain: float(np.mean(vals)) for terrain, vals in terrain_returns.items()}
        terrain_std = {terrain: float(np.std(vals)) for terrain, vals in terrain_returns.items()}

        mean_return_per_terrain = np.array(list(terrain_means.values()), dtype=np.float32)
        overall_mean = float(np.mean(mean_return_per_terrain))
        worst_terrain_return = float(np.min(mean_return_per_terrain))
        std_across_terrains = float(np.std(mean_return_per_terrain))
        robustness_score = float(overall_mean - self.eval_robustness_lambda * std_across_terrains)

        summary_row = {
            'Date': self.date,
            'Design_Counter': int(self.design_counter),
            'Episode_Counter': int(self.episode_counter),
            'Scale_Head': int(SnakeEnv.get_current_design()[0]),
            'Scale_Body': int(SnakeEnv.get_current_design()[1]),
            'Scale_Tail': int(SnakeEnv.get_current_design()[2]),
            'Eval_Episodes_Per_Terrain': int(self.eval_episodes_per_terrain),
            'Overall_Mean_Return': overall_mean,
            'Worst_Terrain_Return': worst_terrain_return,
            'Std_Across_Terrain_Means': std_across_terrains,
            'Robustness_Score': robustness_score,
        }

        for terrain in self.terrain_sequence:
            summary_row[f'{terrain}_Mean_Return'] = terrain_means[terrain]
            summary_row[f'{terrain}_Std_Return'] = terrain_std[terrain]

        results_dir = 'results_bazyli'
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
        }
        detail_json_path = os.path.join(
            results_dir,
            f'{self.date}_Design{self.design_counter}_ep{self.episode_counter}_eval_summary.json'
        )
        with open(detail_json_path, 'w') as f:
            json.dump(detail_payload, f, indent=2)

        print('Evaluation summary:', summary_row)
       
    def save_networks(self):
        """ Saves the networks on the disk.
        """
         # TODO: Edit this to store more efficiently

        results_dir = 'results_bazyli'
        os.makedirs(results_dir, exist_ok=True)

        torch.save(self.rl_alg._ind_policy, os.path.join(results_dir, 'ind_policy_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter)))
        torch.save(self.rl_alg._ind_qf1, os.path.join(results_dir, 'ind_qf1_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter)))
        torch.save(self.rl_alg._ind_qf2, os.path.join(results_dir, 'ind_qf2_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter)))
        torch.save(self.rl_alg._ind_qf1_target, os.path.join(results_dir, 'ind_qf1_tar_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter)))
        torch.save(self.rl_alg._ind_qf2_target, os.path.join(results_dir, 'ind_qf2_tar_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter)))


        torch.save(self.rl_alg._pop_policy, os.path.join(results_dir, 'pop_policy_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter)))
        torch.save(self.rl_alg._pop_qf1, os.path.join(results_dir, 'pop_qf1_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter)))
        torch.save(self.rl_alg._pop_qf2, os.path.join(results_dir, 'pop_qf2_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter)))
        torch.save(self.rl_alg._pop_qf1_target, os.path.join(results_dir, 'pop_qf1_tar_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter)))
        torch.save(self.rl_alg._pop_qf2_target, os.path.join(results_dir, 'pop_qf2_tar_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter)))

        
        metadata = {
            'design_counter': self.design_counter,
            'episode_counter': self.episode_counter,
            'optimized_params': getattr(self, 'optimized_params', None) 
        }

        with open(os.path.join(results_dir, f'{self.date}_Design{self.design_counter}_ep{self.episode_counter}_metadata_carpet.json'), 'w') as f:
            json.dump(metadata, f)

        self.save_replay(os.path.join(results_dir, f'replay_{self.date}_Design{self.design_counter}_carpet.pt'))

        print(f"saved networks for design {self.design_counter} and episode {self.episode_counter}")    

    def load_networks(self, base_path, checkpoint_prefix):
        self.rl_alg._ind_policy.load_state_dict(torch.load(
            f'{base_path}/ind_policy_{checkpoint_prefix}_carpet.pt'
        ).state_dict())
        self.rl_alg._ind_qf1.load_state_dict(torch.load(
            f'{base_path}/ind_qf1_{checkpoint_prefix}_carpet.pt'
        ).state_dict())
        self.rl_alg._ind_qf2.load_state_dict(torch.load(
            f'{base_path}/ind_qf2_{checkpoint_prefix}_carpet.pt'
        ).state_dict())
        self.rl_alg._ind_qf1_target.load_state_dict(torch.load(
            f'{base_path}/ind_qf1_tar_{checkpoint_prefix}_carpet.pt'
        ).state_dict())
        self.rl_alg._ind_qf2_target.load_state_dict(torch.load(
            f'{base_path}/ind_qf2_tar_{checkpoint_prefix}_carpet.pt'
        ).state_dict())

        self.rl_alg._pop_policy.load_state_dict(torch.load(
            f'{base_path}/pop_policy_{checkpoint_prefix}_carpet.pt'
        ).state_dict())
        self.rl_alg._pop_qf1.load_state_dict(torch.load(
            f'{base_path}/pop_qf1_{checkpoint_prefix}_carpet.pt'
        ).state_dict())
        self.rl_alg._pop_qf2.load_state_dict(torch.load(
            f'{base_path}/pop_qf2_{checkpoint_prefix}_carpet.pt'
        ).state_dict())
        self.rl_alg._pop_qf1_target.load_state_dict(torch.load(
            f'{base_path}/pop_qf1_tar_{checkpoint_prefix}_carpet.pt'
        ).state_dict())
        self.rl_alg._pop_qf2_target.load_state_dict(torch.load(
            f'{base_path}/pop_qf2_tar_{checkpoint_prefix}_carpet.pt'
        ).state_dict())

        print(f"loaded networks from checkpoint: {checkpoint_prefix}")

        metadata_path = f'{base_path}/{checkpoint_prefix}_metadata_carpet.json'

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.design_counter = metadata['design_counter']
            self.episode_counter = metadata['episode_counter']
            self.optimized_params = metadata.get('optimized_params', None)
            print(f"restored design_counter={self.design_counter}, episode_counter={self.episode_counter}")
        else:
            print("no metadata file found; counters not restored.")

        replay_path = f'{base_path}/replay_{checkpoint_prefix.split("_ep")[0]}_carpet.pt'
        if os.path.exists(replay_path):
            self.load_replay(replay_path)
            print("Replay contains", self.replay._individual_buffer._size, "steps")
        else:
            print("no replay buffer found.")



    def _serialize_replay_buffer(self, buffer):
        return {
            "observations": buffer._observations,
            "actions": buffer._actions,
            "rewards": buffer._rewards,
            "terminals": buffer._terminals,
            "next_observations": buffer._next_obs,
            "env_infos": {key: value for key, value in buffer._env_infos.items()},
            "env_info_keys": list(buffer._env_info_keys),
            "_top": buffer._top,
            "_size": buffer._size,
            "_replace": buffer._replace,
            "_max_replay_buffer_size": buffer._max_replay_buffer_size,
            "_observation_dim": buffer._observation_dim,
            "_action_dim": buffer._action_dim,
        }

    def _restore_replay_buffer(self, buffer, data):
        buffer._observations = data["observations"]
        buffer._actions = data["actions"]
        buffer._rewards = data["rewards"]
        buffer._terminals = data["terminals"]
        buffer._next_obs = data["next_observations"]
        buffer._env_infos = data.get("env_infos", buffer._env_infos)
        buffer._env_info_keys = data.get("env_info_keys", list(buffer._env_infos.keys()))
        buffer._top = data["_top"]
        buffer._size = data["_size"]
        buffer._replace = data.get("_replace", buffer._replace)
        buffer._max_replay_buffer_size = data.get("_max_replay_buffer_size", buffer._max_replay_buffer_size)
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
                # stored the individual replay buffer.
                self._restore_replay_buffer(self.replay._individual_buffer, data)
                print(f"loaded legacy replay buffer from {filepath} with {self.replay._individual_buffer._size} samples")
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
        rewardDF['Terrain'] = [SnakeEnv.get_current_terrain()] * len(self.timesteps)
        design = SnakeEnv.get_current_design()
        rewardDF['Terrain_ID'] = [self.episode_counter % len(self.terrain_sequence)] * len(self.timesteps)
        rewardDF['Scale_Head'] = [int(design[0])] * len(self.timesteps)
        rewardDF['Scale_Body'] = [int(design[1])] * len(self.timesteps)
        rewardDF['Scale_Tail'] = [int(design[2])] * len(self.timesteps)

        # log state variablesmotor_and_coadaptation/CoadaptationCode/train_coadapt.py
        for motor_idx, motor_actions in enumerate(self.actionList):
            rewardDF[f'Motor{motor_idx + 1}_Action'] = motor_actions
      

        # log state variables
        rewardDF['X_State'] = self.stateList[0]
        rewardDF['Y_State'] = self.stateList[1]
        rewardDF['Z_State'] = self.stateList[2]
        #rewardDF['X_Heading'] = self.stateList[3]
        rewardDF['Y_Heading'] = self.stateList[3]
        #rewardDF['Z_Heading'] = self.stateList[5]
        rewardDF['Motor1_Pos'] = self.stateList[4]
        rewardDF['Motor2_Pos'] = self.stateList[5]
        rewardDF['Motor3_Pos'] = self.stateList[6]
        rewardDF['Motor4_Pos'] = self.stateList[7]
        rewardDF['Motor5_Pos'] =  self.stateList[8]
        rewardDF['Motor6_Pos'] =  self.stateList[9]
        if len(self.stateList) > 10:
            rewardDF['Motor7_Pos'] = self.stateList[10]
        if len(self.stateList) > 11:
            rewardDF['Design_1'] = self.stateList[11]
        if len(self.stateList) > 12:
            rewardDF['Design_2'] = self.stateList[12]
        if len(self.stateList) > 13:
            rewardDF['Design_3'] = self.stateList[13]

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
    base_path = "/home/bazyli/Desktop/Snake Robot Project/Repo/SnakeRobot/CoadaptationCode/results_bazyli"
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
