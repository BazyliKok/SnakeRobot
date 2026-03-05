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
        self._episode_length = 175 # number of timesteps per episode
        self.episode_counter = None
        self.episodes_before_training = 0 #4 # number of episodes before training to fill the replay buffer
        self.episode_iterations = 30 # number of episodes per design
        self.design_cylces = 20 # total number of design cycles

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
        
        

    def run(self, stopEvent):
        """ Runs the Fast Evolution through Actor-Critic RL algorithm.

        First the initial design loop is executed in which the rl-algorithm
        is exeuted on the initial designs. Then the design-optimization
        process starts.
        It is possible to have different numbers of iterations for initial
        designs and the design optimization process.
        """
        self.stateList = []
        self.actionList = [[] for i in range(0,7)] #was 6
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

        # setting up files and file names
        self.date = datetime.now().strftime("%Y_%m_%d")
        name = "Rewards_Design{}_carpet".format(str(self.design_counter))
        self.filename = self.date+name
        name = "Losses_Design{}_carpet".format(str(self.design_counter))
        self.lossFilename = self.date+name

        ptu.set_gpu_mode(True) # making sure to use GPU
       
        #self.optimized_params = [3.5904986579519607, 1.7847799600014809, 3.584383159576394, 3.3680278532796404, 1.2309833188420878, 0.5229256834552103, 3.250148806721444]
        
        # determine what to do based on design cycle currently on
        if self.design_counter < self.num_init_designs: # if not done with initial design loop
            self.initial_design_loop() # run another initial design loop
            print(f'design counter at {self.design_counter}')
            if self.design_counter == self.num_init_designs: # if on last initial design cycle
                # last initial design
                self.first_train_op() # run design
            stopEvent.set() # end thread
            return
        
        
        elif self.design_counter < self.design_cylces: # if still on all other designs
            self.train_loop()
            stopEvent.set() # end thread
            return

        else: # have reached end of training
            # can run end sequence here
            pass 
    

    def collect_training_experience(self):
            """ Collect training data.

            This function executes a single episode in the environment using the
            exploration strategy/mechanism and the policy.
            The data, i.e. state-action-reward-nextState, is stored in the replay
            buffer.

            """

            self.stateList = []
            self.actionList = [[] for i in range(0,7)] # was 6
            self.timestepRewards = []
            self.cumulativeRewards = []
            self.epList = []
            self.timesteps = []

            # reset environment
            terrain = self.terrain_sequence[self.episode_counter % len(self.terrain_sequence)]
            SnakeEnv.set_current_terrain(terrain)
            print(f"CURRENT TERRAIN: {terrain}. Place robot on this terrain before continuing reset.")
            state, info = self.env.reset()
            state_dim = len(state)
            self.stateList = [[] for _ in range(state_dim)]
            steps = 0
            episodeRewards = 0
            episodeContRewards = []
            Done = False
        
            # get policies
            self.policy = self.rl_alg.get_policy_network(self.networks['individual']) #get policy here
            self.pop_policy = self.rl_alg.get_policy_network(self.networks['population']) #get policy here

            currDesign = SnakeEnv.get_current_design()
            
            while not (Done) and steps <= self._episode_length:
                start = time.time()
                
                self.timesteps.append(steps)


                steps += 1
                print(f'Step: {steps}')
                #state = torch.tensor(state)
                #state = state.to(torch.float32)

                
                # exploration vs exploitation
                #TODO:change to >
                if self.currEp >= self.episodes_before_training : # can start training, exploitation
                    action,_ = self.policy.get_action(state, deterministic=False) 
                else: # purely exploring
                    #action, _= self.pop_policy.get_action(state, deterministic=False)
                    action = np.random.uniform(-1,1, size=7) # this is for early designs


                for i in range(0,6): # for data logging purpose
                    self.actionList[i].append(action[i])
                
        
                next_state, reward, terminated, truncated, info = self.env.step(action) # step the action, note: reward is scaled in environment

                if steps > self._episode_length:
                    SnakeEnv.disableMotorTorque() # stop motors when reach end of an episode
                    print('disabled torque')
                
                episodeRewards += reward # accumulate rewards here to track for comparison
        
                # log rewards
                self.timestepRewards.append(reward)
                self.cumulativeRewards.append(episodeRewards)
                self.epList.append(self.currEp) # to make note of what episode we are on
                for i in range(len(state)): #was 17
                    self.stateList[i].append(state[i])


             

                Done = terminated # can check here for terminated and truncated
                terminal = np.array([Done]) # turn into array for replay buffer
                reward = np.array([reward])
                
                # add replay sample

                print(f'action shape: {action.shape}')
                self.replay.add_sample(observation=state, action=action, reward=reward, next_observation=next_state,
                   terminal=terminal, env_info={})

                state = next_state # set state for next iteration
                
             
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

        # Design Optimization
        print(f'design counter at {self.design_counter}')
        if self.design_counter >= self.num_init_designs:
            self._data_design_type = 'Optimized'
            q_network = self.rl_alg.get_q_network(self.networks['population'])
            policy_network = self.rl_alg.get_policy_network(self.networks['population'])
            self.cost, self.optimized_params = self.do_alg.optimize_design(design=self.optimized_params, q_network=q_network, policy_network=policy_network)
            self.optimized_params = list(self.optimized_params)
            self.design_counter += 1 # another design
            print('NEW DESIGN PARAMETERS: ',self.optimized_params)
            print('COST: ', self.cost)
        #else: # randomize next design
        #    self._data_design_type = 'Random'
        #    self.optimized_params = SnakeEnv.get_random_design()
        #    self.optimized_params = list(self.optimized_params)

        
        self.design_counter += 1 # another design
        
            
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
        # evaluate policy
        self.evaluate_policy()

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
        for _ in range(self.episode_iterations):
            self.currEp = _
            print('in initial design loop')
            self.train_single_iteration()

            print(f'range {range(self.episode_counter, self.episode_iterations)}')
        self.design_counter+= 1
        
        return
          
    def evaluate_policy(self):
        """ Evaluates the current deterministic policy.

        Evaluates the current policy in the environment by unrolling a single
        episode in the environment.
        The achieved cumulative reward is logged.
        """
        # can add a policy rollout here
        pass
       
    def save_networks(self):
        """ Saves the networks on the disk.
        """
         # TODO: Edit this to store more efficiently
       
        torch.save(self.rl_alg._ind_policy, 'results/ind_policy_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._ind_qf1, 'results/ind_qf1_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._ind_qf2, 'results/ind_qf2_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._ind_qf1_target, 'results/ind_qf1_tar_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._ind_qf2_target, 'results/ind_qf2_tar_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter))


        torch.save(self.rl_alg._pop_policy, 'results/pop_policy_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._pop_qf1, 'results/pop_qf1_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._pop_qf2, 'results/pop_qf2_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._pop_qf1_target, 'results/pop_qf1_tar_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter))
        torch.save(self.rl_alg._pop_qf2_target, 'results/pop_qf2_tar_{}_Design{}_ep{}_carpet.pt'.format(self.date, self.design_counter, self.episode_counter))
        
        metadata = {
            'design_counter': self.design_counter,
            'episode_counter': self.episode_counter,
            'optimized_params': getattr(self, 'optimized_params', None) 
        }

        with open(f'results/{self.date}_Design{self.design_counter}_ep{self.episode_counter}_metadata_carpet.json', 'w') as f:
            json.dump(metadata, f)

        self.save_replay(f'results/replay_{self.date}_Design{self.design_counter}_carpet.pt')

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

        print("loaded networks from checkpoint: {checkpoint_prefix}")

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



    def save_replay(self, filepath):
        """Save replay buffer content to disk."""

        try:
            buf = self.replay._individual_buffer
            data = {
                "observations": buf._observations,
                "actions": buf._actions,
                "rewards": buf._rewards,
                "terminals": buf._terminals,
                "next_observations": buf._next_obs,
                "_top": buf._top,
                "_size": buf._size,
            }
            torch.save(data, filepath)
            print(f"saved replay buffer to {filepath}")
        except Exception as e:
            print(f"failed to save replay buffer: {e}")

    def load_replay(self, filepath):
        """Load replay buffer content from disk."""
        try:
            buf = self.replay._individual_buffer
            data = torch.load(filepath)

            buf._observations = data["observations"]
            buf._actions = data["actions"]
            buf._rewards = data["rewards"]
            buf._terminals = data["terminals"]
            buf._next_obs = data["next_observations"]
            buf._top = data["_top"]
            buf._size = data["_size"]

            print(f"loaded replay buffer from {filepath} with {buf._size} samples")
        except Exception as e:
            print(f"failed to load replay buffer: {e}")




    def logData(self):

        xPositionList, yPositionList = SnakeEnv.returnOptiXList()
        min_len = min(len(self.timesteps), len(xPositionList))

        # trim all lists to the same length
        self.timesteps = self.timesteps[:min_len]
        self.timestepRewards = self.timestepRewards[:min_len]
        self.cumulativeRewards = self.cumulativeRewards[:min_len]
        xPositionList = xPositionList[:min_len]
        yPositionList = yPositionList[:min_len]
        self.epList = self.epList[:min_len]

        for i in range(7): #was 6
            self.actionList[i] = self.actionList[i][:min_len]
        for i in range(len(self.stateList)):
            self.stateList[i] = self.stateList[i][:min_len]
        rewardDF = pd.DataFrame()

        rewardDF['Episode'] = [self.episode_counter]*176
        rewardDF['Timestep'] = self.timesteps
        rewardDF['X_Position']= xPositionList # added this, need to see if it works
        rewardDF['Y_Position']= yPositionList # added this, need to see if it works
        rewardDF['Rewards'] = self.timestepRewards
        rewardDF['Cumulative_Rewards'] = self.cumulativeRewards
        rewardDF['Terrain'] = [SnakeEnv.get_current_terrain()] * len(self.timesteps)
        design = SnakeEnv.get_current_design()
        rewardDF['Scale_Head'] = [int(design[0])] * len(self.timesteps)
        rewardDF['Scale_Body'] = [int(design[1])] * len(self.timesteps)
        rewardDF['Scale_Tail'] = [int(design[2])] * len(self.timesteps)

        # log state variablesmotor_and_coadaptation/CoadaptationCode/train_coadapt.py
        rewardDF['Motor1_Action'] = self.actionList[0]
        rewardDF['Motor2_Action'] = self.actionList[1]
        rewardDF['Motor3_Action'] = self.actionList[2]
        rewardDF['Motor4_Action'] = self.actionList[3]
        rewardDF['Motor5_Action'] = self.actionList[4]
        rewardDF['Motor6_Action'] = self.actionList[5]
      

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
            rewardDF['Design_1'] = self.stateList[10]
        if len(self.stateList) > 11:
            rewardDF['Design_2'] = self.stateList[11]
        if len(self.stateList) > 12:
            rewardDF['Design_3'] = self.stateList[12]

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
    base_path = "/home/liza/SnakeRobot/CoadaptationCode/results"
    #change name
    checkpoint_prefix = "2025_06_03_Design0_ep30"

    #set to false if new training starts
    resuming_from_checkpoint = True  

    if resuming_from_checkpoint:
        trainingObj.episode_counter = 30
        trainingObj.load_networks(base_path, checkpoint_prefix)
    else:
        trainingObj.episode_counter = 0
        print("Starting fresh: episode_counter set to 0")

    # run threads as before
    motorThread = threading.Thread(target=trainingObj.motorPos, args=(stopEvent,)) 
    optiThread = threading.Thread(target=trainingObj.optiPos, args=(stopEvent,))
    trainingloopThread = threading.Thread(target=trainingObj.run, args=(stopEvent,))

    motorThread.start()
    optiThread.start() 
    trainingloopThread.start()
    trainingloopThread.join()
