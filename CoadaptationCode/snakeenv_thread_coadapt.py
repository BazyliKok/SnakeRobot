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
import os
import copy
from datetime import datetime

gymnasium.envs.register(
    id = "SnakeRobot",
    entry_point = f"{__name__}:SnakeEnv",
    max_episode_steps = 150,  # maybe come back and change
    reward_threshold = 1000,
    
)
global optiPos, motorPos

class SnakeEnv(gymnasium.Env):
    # static variables so can be accessed between static and non static methods
    optiPosition = []
    motorPosition = []
    optiXTrack = []
    optiYTrack = []
    prevPos = [0,0,0]
    optiRelPos = []
    motors = motorssynced.MotorsSynced()
    opti = optitrack.Optitrack()
    motorLock = threading.Lock()
    optiLock = threading.Lock()
    starting_angle = None

    bla = time.time()


    '''
       Robot has 7 motors and 8 snake segments
       Action Space: 7
       Observation Space: 12 from snake robot + 7 from design 
    '''

    # setting up design framework
    #TODO: make it 1.80 for 1 segment
    # Design vector encodes the scale type used on [head, body, tail].
    # 0 = TPU with spikes, 1 = TPU without spikes,
    # 2 = PLA with spikes, 3 = PLA without spikes.
    current_design = [0, 0, 0]
    current_terrain = 'floor'
    scale_types = {
        0: 'TPU_SPIKES',
        1: 'TPU_NO_SPIKES',
        2: 'PLA_SPIKES',
        3: 'PLA_NO_SPIKES',
    }
    terrains = ['floor', 'carpet', 'cardboard', 'artificial_grass']

    #design_parameter_bounds = [(.45, 3.60)] # the bounds of how small and large a variable can be 

    # Discrete bounds for [head, body, tail] scale type ids.
    design_parameter_bounds = [(0,3), (0,3), (0,3)]

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

    config_numpy = np.array(current_design)
    
    
    design_dims =  list(range(14-len(current_design), 15)) # either 14or 15
    print('design dimensions!', design_dims)
    
    def __init__(self):
    
        self.rewardScale = 100 # scale rewards
        self.motorMin = 1422 #1500 #1422
        self.motorMax = 2674 #2500 #2673
        
       

        base_obs_dim = 11
        design_dim = len(SnakeEnv.config_numpy)
        obs_dim = base_obs_dim + design_dim
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype='float32'
        )
        
        self.action_space = spaces.Box(low=self.motorMin, high=self.motorMax, shape=(7,), dtype='float32') # continuous action space
        #self.action_space = spaces.Box(low=self.motorMin, high=self.motorMax, shape=(18,), dtype='float32')

        self.targetPositionX = -100.0 # position that can't be reached, think about changing or getting rid of this

               
        # init other things
        # moved these class declarations to static
        self.motors = motorssynced.MotorsSynced()
        self.opti = optitrack.Optitrack()
        #self.opti.optiTrackInit()

        self.currPosition =[]
        self.newAction = []

        self.reward = 0
        self.prevDist = 0
        self.prevPos = 0 

        self.distList = []
        self.rewardList = []
        self.xPosList = []
        self.i = 0

        # data frame for logging data
        self.df = pd.DataFrame(columns=['Action Sent','Opti Position', 'Motor Position','Reward'])

        # set up files
        self.filename = "Training_" + datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        
     
    def step(self, action):

        num_timesteps = 1  
        assert len(action) == 7 * num_timesteps, f"Action space must now be {7 * num_timesteps} values ({7} motors × {num_timesteps} timesteps)."

        # Split action into multiple consecutive timesteps
        actions = [action[i * 7: (i + 1) * 7] for i in range(num_timesteps)]

        for sub_action in actions:
            actionForMotors = self.denormalizeAction(sub_action)
            print(actionForMotors)
            self.writeAction(actionForMotors)

        # Wait 0.5s before retrieving the next batch of actions
        for i in range(3):
            nextObs = self._get_obs()
        #time.sleep(.2)

        for i in range(5):
            nextObs = self._get_obs()
            #print(nextObs)
        tmp_pos = copy.deepcopy(nextObs)
        # nextObs[0] = nextObs[0] - self._prev_obs[0]
        nextObs[1] = nextObs[1] - self._prev_obs[1]
        nextObs[2] = nextObs[2] - self._prev_obs[2]

        max_wait = 50
        wait_i = 0
        eps = 1e-3
        while abs(nextObs[0] - self._prev_obs[0]) < eps and abs(nextObs[1]) < eps and wait_i < max_wait:
            nextObs = self._get_obs()
            print(nextObs)
            tmp_pos = copy.deepcopy(nextObs)
            # nextObs[0] = nextObs[0] - self._prev_obs[0]
            nextObs[1] = nextObs[1] - self._prev_obs[1]
            nextObs[2] = nextObs[2] - self._prev_obs[2]
            print("im here")
            print(nextObs[0], nextObs[1])
            wait_i += 1

        self._prev_obs = tmp_pos

        # Log global positions
        SnakeEnv.optiXTrack.append(SnakeEnv.optiRelPos[0])  # global x position of robot
        SnakeEnv.optiYTrack.append(SnakeEnv.optiRelPos[2])  # global y position of robot

        # extract X position
        currXPos = nextObs[0]  # opti X position of the robot

        # # reward forward movement
        # reward = max([currXPos*0.05,0.]) * (.3- abs(nextObs[3]))  #(currXPos - self.prevPos)
        #print("global pos")
        #print(global_pos)

        max_distance = 40 #self.targetPositionX - ( 20) 
        distance = abs(self.targetPositionX - currXPos)
        print("reward {}".format(np.exp(1 - (distance / max_distance))))
        reward = np.exp(1 - (distance / max_distance)) +  (.3- abs(nextObs[3]))

        # check if the goal is reached
        terminated = currXPos < self.targetPositionX

        truncated = False
        info = {'info': 0}

        print(f"Reward: {reward}")

        # Log data
        self.df.loc[len(self.df.index)] = [actionForMotors, nextObs[0:7], nextObs[7:-1], reward]

        # update previous position and action
        self.prevPos = currXPos

        # add design info to observation

        nextObs = np.append(nextObs, SnakeEnv.config_numpy)
        print(f"Observation: {nextObs}")

        return np.array(nextObs, dtype=np.float32), reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        # returns: observation of the initial state
        print('in reset')
       
        
        super().reset(seed=seed)  # this is needed for cutom environments according to AI Gym

        input('Reset robot then press a button to continue') # used to pause to reset robot position
   


        SnakeEnv.motorLock.acquire()
        SnakeEnv.motors.setMotorSpeed() # set speed here so if motor torques and reset power the speed gets reset
        time.sleep(.5)
        SnakeEnv.motorLock.release()
        # time.sleep(.5)
        print('motor speeds set')


        # choose starting position of robot motors
        #startPos = random.sample(range(self.motorMin, self.motorMax), 7)
        startPos = [2048, 2048, 2048, 2048, 2048, 2048, 2048]
        self.writeAction(startPos)
        SnakeEnv.disableMotorTorque()
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
        observation = self._get_obs(initial=True)
        # observation[0] = 0.
        observation[1] = 0.
        observation[2] = 0.
        SnakeEnv.enableMotorTorque()
        # DO NOT UNCOMMENT IN

        print('Observation: ', observation)
        self.starting_position = observation[0]
        self.prevPos = observation[0] # x position of observation

        self._prev_obs = copy.deepcopy(observation)


        info = {'info': 0}

        observationfull = np.append(observation, SnakeEnv.config_numpy)

        print('full observation', observationfull)
        return (np.array(observationfull, dtype=np.float32), info)

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

    def _get_obs(self, initial=False):
        # read agent x,y,z,etc and target goal pos
        # return {"agent": self._agent_location, "target": self._target_location}
        
        self.agentPos = self.getPosition(initial)
       
        return [*self.agentPos]
    
    def getPosition(self, initial):
       
        # motorPos = self.motors.readPos()
        # optiPos = self.opti.optiTrackGetPos() # currently returning x, y, z
        # self.currPosition = [*optiPos, *motorPos]  
        
 
        SnakeEnv.motorLock.acquire()
        SnakeEnv.optiLock.acquire()

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
        SnakeEnv.motors.writePos(posTo)
        SnakeEnv.motorLock.release()

        time.sleep(.3) # sleep to allow motors to get to position


    def getTorque(self):
        motorTor = self.motors.readTorque(self.motorLock)
        return motorTor
    
    def denormalizeAction(self, action):
        
        motorMax = self.motorMax
        motorMin = self.motorMin

        mapping = interp1d([-1, 1], [motorMin, motorMax])
        mappedList = [int(mapping(i)) for i in action]

        return mappedList

    '''
        The following methods are static so they can be accessed from outside environment wrapper to edit parameters with threading 
    '''
    @staticmethod
    def passLocksToEnv(oLock, mLock):
        # function to pass locks into this environment
        SnakeEnv.optiLock = oLock
        SnakeEnv.motorLock = mLock
        return
    
    @staticmethod
    def optiPos():
        SnakeEnv.optiLock.acquire()
        SnakeEnv.optiRelPos, heading = SnakeEnv.opti.optiTrackGetPos()
        #print(time.time() - SnakeEnv.bla)
        SnakeEnv.optiPosition = [*SnakeEnv.optiRelPos, *heading]
        #print(SnakeEnv.optiPosition)
        SnakeEnv.optiLock.release()
        time.sleep(.001) # changed from .008
        #SnakeEnv.bla = time.time()
        return
    
    @staticmethod
    def motorPos():
        SnakeEnv.motorLock.acquire()
        SnakeEnv.motorPosition = SnakeEnv.motors.readPos()
        # print('In thread', SnakeEnv.motorPosition)
        SnakeEnv.motorLock.release()
        time.sleep(.001)
        return
    
    @staticmethod
    def returnOptiXList():
        return SnakeEnv.optiXTrack, SnakeEnv.optiYTrack
    

    @staticmethod
    def disableMotorTorque():
        SnakeEnv.motorLock.acquire()
        SnakeEnv.motorPosition = SnakeEnv.motors.disableTorque()
        SnakeEnv.motorLock.release()
        #time.sleep(.005)
        return   

    @staticmethod
    def enableMotorTorque():
        SnakeEnv.motorLock.acquire()
        SnakeEnv.motorPosition = SnakeEnv.motors.enableTorque()
        SnakeEnv.motorLock.release()
        #time.sleep(.005)
        return   

    @staticmethod
    def set_new_design(design):
        # Keep design ids in [0..3] and integer-coded.
        rounded = [int(np.clip(np.round(v), 0, 3)) for v in design]
        SnakeEnv.current_design = rounded
        SnakeEnv.config_numpy = np.array(rounded, dtype=np.float32)

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
        optimized_params = np.random.uniform(low=SnakeEnv.design_parameter_bounds[0][0], high=SnakeEnv.design_parameter_bounds[0][1], size=7)
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
        

    
