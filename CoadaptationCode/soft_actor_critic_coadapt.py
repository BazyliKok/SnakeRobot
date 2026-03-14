'''
This is from co adaptation framework: https://github.com/ksluck/Coadaptation/blob/master/RL/soft_actor.py
'''
from rlkit.torch.sac.policies import TanhGaussianPolicy
# from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import ConcatMlp
import numpy as np
from rl_algorithm import RLAlgorithm
# from rlkit.torch.sac.sac import SACTrainer
from SACTrainer import SACTrainer
import rlkit.torch.pytorch_util as ptu
import torch
import utils
import torch.optim as optim

class SoftActorCriticCoadapt(RLAlgorithm):
    
    def __init__(self, env, replay, networks):
        """ Bascally a wrapper class for SAC from rlkit.

        Args:
            config: Configuration dictonary
            env: Environment
            replay: Replay buffer
            networks: dict containing two sub-dicts, 'individual' and 'population'
                which contain the networks.

        """
        ptu.set_gpu_mode(False)
        super().__init__(env, replay, networks)
        ptu.set_gpu_mode(False) 

        # define networks for individual
        self._ind_qf1 = networks['individual']['qf1']
        self._ind_qf2 = networks['individual']['qf2']
        self._ind_qf1_target = networks['individual']['qf1_target']
        self._ind_qf2_target = networks['individual']['qf2_target']
        self._ind_policy = networks['individual']['policy']

        # define networks for policy
        self._pop_qf1 = networks['population']['qf1']
        self._pop_qf2 = networks['population']['qf2']
        self._pop_qf1_target = networks['population']['qf1_target']
        self._pop_qf2_target = networks['population']['qf2_target']
        self._pop_policy = networks['population']['policy']

        # define training parameters
        self._batch_size = 32
        self._nmbr_ind_updates = 100  # cap on individual SAC updates per train call
        self._nmbr_pop_updates = 50  # cap on population SAC updates per train call
        self._sac_trainer_kwargs = dict(
            discount=0.99,
            reward_scale=1.0,
            policy_lr=1E-3,
            qf_lr=1E-3,
            optimizer_class=optim.Adam,
            soft_target_tau=.01,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
            alpha=0.1,
        )

        # set up trainer 
        self._ind_algorithm = self._build_trainer(
            policy=self._ind_policy,
            qf1=self._ind_qf1,
            qf2=self._ind_qf2,
            target_qf1=self._ind_qf1_target,
            target_qf2=self._ind_qf2_target,
        )

        self._pop_algorithm = self._build_trainer(
            policy=self._pop_policy,
            qf1=self._pop_qf1,
            qf2=self._pop_qf2,
            target_qf1=self._pop_qf1_target,
            target_qf2=self._pop_qf2_target,
        )

        

    
    def episode_init(self):
           
        """ Initializations to be done before the first episode.

        In this case basically creates a fresh instance of SAC for the
        individual networks and copies the values of the target network.
        """
        ptu.set_gpu_mode(False)
        self._ind_algorithm = self._build_trainer(
            policy=self._ind_policy,
            qf1=self._ind_qf1,
            qf2=self._ind_qf2,
            target_qf1=self._ind_qf1_target,
            target_qf2=self._ind_qf2_target,
        )

        self._networks['individual'] 

        # comment these back in if want to change reset networks at beginning
        #init = SoftActorCriticCoadapt._create_networks(env=self._env)
        #utils.copy_pop_to_ind(networks_pop=init, networks_ind=self._networks['individual'])

        print('RESET INIT')
        #CHANGE 6/16 try without copying network
        utils.copy_pop_to_ind(networks_pop=self._networks['population'], networks_ind=self._networks['individual'])

    def _build_trainer(self, policy, qf1, qf2, target_qf1, target_qf2):
        return SACTrainer(
            env=self._env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **self._sac_trainer_kwargs,
        )
        
  
    def single_train_step(self, train_ind=True, train_pop=False):
        """
            single step in the training
        """
        ptu.set_gpu_mode(False)
        self.trainQ1losses = []
        self.trainQ2losses = []
        self.trainPolicylosses = []

        self.popQ1losses = [] 
        self.popQ2losses = [] 
        self.popPolicylosses = [] 

        print('IN TRAINING')
        if train_ind:
            self._replay.set_mode('species')
            ind_steps_can_sample = self._replay.num_steps_can_sample()
            ind_updates = min(
                self._nmbr_ind_updates,
                max(1, int(np.ceil(ind_steps_can_sample / self._batch_size)))
            )
            print(f'individual SAC updates: {ind_updates} from {ind_steps_can_sample} replay steps')
            self._ind_algorithm.start_diagnostics_epoch()
            for i in range(ind_updates):
                #print('in ind for')
                batch = self._replay.random_batch(self._batch_size)
                self._ind_algorithm.train(batch)
                #print('trained')
            
            traindata = self._ind_algorithm.get_diagnostics()
            self.trainQ1losses.append(traindata['QF1 Loss'])
            self.trainQ2losses.append(traindata['QF2 Loss'])
            self.trainPolicylosses.append(traindata['Policy Loss'])
            self._ind_algorithm.end_epoch(1)

        if train_pop:
            self._replay.set_mode('population')
            pop_steps_can_sample = self._replay.num_steps_can_sample()
            pop_updates = min(
                self._nmbr_pop_updates,
                max(1, int(np.ceil(pop_steps_can_sample / self._batch_size)))
            )
            print(f'population SAC updates: {pop_updates} from {pop_steps_can_sample} replay steps')
            self._pop_algorithm.start_diagnostics_epoch()
            for i in range(pop_updates):
                #print('in pop for')
                batch = self._replay.random_batch(self._batch_size)
                self._pop_algorithm.train(batch)

            traindataPop = self._pop_algorithm.get_diagnostics()
            self.popQ1losses.append(traindataPop['QF1 Loss'])
            self.popQ2losses.append(traindataPop['QF2 Loss'])
            self.popPolicylosses.append(traindataPop['Policy Loss'])
            self._pop_algorithm.end_epoch(1)
        else:
            self.popQ1losses.append(0)
            self.popQ2losses.append(0)
            self.popPolicylosses.append(0)

        return self.trainQ1losses, self.trainQ2losses, self.trainPolicylosses, self.popQ1losses, self.popQ2losses, self.popPolicylosses
    



    @staticmethod
    def create_networks(env):
        """ Creates all networks necessary for SAC.

        These networks have to be created before instantiating this class and
        used in the constructor.

        Returns:
            A dictonary which contains the networks.
        """
        ptu.set_gpu_mode(False)
        network_dict = {
            'individual' : SoftActorCriticCoadapt._create_networks(env=env),
            'population' : SoftActorCriticCoadapt._create_networks(env=env),    
            }
        
        return network_dict
  
   
    
    @staticmethod
    def _create_networks(env):
        """ Creates all networks necessary for SAC.

        These networks have to be created before instantiating this class and
        used in the constructor.

        Args:
            config: A configuration dictonary.

        Returns:
            A dictonary which contains the networks.
        """
        obs_dim = int(np.prod(env.observation_space.shape)) # will need to check if this works
        action_dim = int(np.prod(env.action_space.shape))
        net_size = 256
        hidden_sizes = [net_size] * 3
        # hidden_sizes = [net_size, net_size, net_size]

        ptu.set_gpu_mode(False)
        device = torch.device('cuda:0')
        # could try different networks
        qf1 = ConcatMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        qf2 = ConcatMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        qf1_target = ConcatMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        qf2_target = ConcatMlp(
            hidden_sizes=hidden_sizes,
            input_size=obs_dim + action_dim,
            output_size=1,
        ).to(device=ptu.device)
        #TODO: check if action_dim is 18
        policy = TanhGaussianPolicy(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim,
            action_dim=action_dim,
        ).to(device=ptu.device)
        # SAC target critics should start as exact copies of the online critics.
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())

        print("obs dim number", obs_dim)


        clip_value = 1.0
        for p in qf1.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in qf2.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))
        for p in policy.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

        return {'qf1' : qf1, 'qf2' : qf2, 'qf1_target' : qf1_target, 'qf2_target' : qf2_target, 'policy' : policy}

    @staticmethod
    def get_q_network(networks):
        """ Returns the q network from a dict of networks.

        This method extracts the q-network from the dictonary of networks
        created by the function create_networks.

        Args:
            networks: Dict containing the networks.

        Returns:
            The q-network as torch object.
        """
        return networks['qf1']

    @staticmethod
    def get_policy_network(networks):
        """ Returns the policy network from a dict of networks.

        This method extracts the policy network from the dictonary of networks
        created by the function create_networks.

        Args:
            networks: Dict containing the networks.

        Returns:
            The policy network as torch object.
        """
        return networks['policy']
