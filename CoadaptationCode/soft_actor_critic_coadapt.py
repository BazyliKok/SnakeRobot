'''
This is from co adaptation framework: https://github.com/ksluck/Coadaptation/blob/master/RL/soft_actor.py
'''
from rlkit.torch.sac.policies import TanhGaussianPolicy
# from rlkit.torch.sac.sac import SoftActorCritic
from rlkit.torch.networks import ConcatMlp
import numpy as np
import os
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

        self._terrain_model_mode = str(
            networks.get(
                'terrain_model_mode',
                os.getenv('SNAKE_TERRAIN_MODEL_MODE', 'separate'),
            )
        ).strip().lower()
        if self._terrain_model_mode not in ('separate', 'shared'):
            raise ValueError(
                "SNAKE_TERRAIN_MODEL_MODE must be either 'separate' or 'shared'."
            )

        self._terrain_names = list(networks.get('terrain_names') or ['default'])
        self._active_terrain = self._terrain_names[0]

        if self._terrain_model_mode == 'separate':
            networks.setdefault('individual_by_terrain', {})
            networks.setdefault('population_by_terrain', {})
            for terrain in self._terrain_names:
                if terrain not in networks['individual_by_terrain']:
                    networks['individual_by_terrain'][terrain] = self._create_networks(env=self._env)
                if terrain not in networks['population_by_terrain']:
                    networks['population_by_terrain'][terrain] = self._create_networks(env=self._env)
            networks['individual'] = networks['individual_by_terrain'][self._active_terrain]
            networks['population'] = networks['population_by_terrain'][self._active_terrain]
        else:
            networks.setdefault('individual_by_terrain', {'shared': networks['individual']})
            networks.setdefault('population_by_terrain', {'shared': networks['population']})
            self._active_terrain = 'shared'

        self._individual_networks_by_terrain = networks['individual_by_terrain']
        self._population_networks_by_terrain = networks['population_by_terrain']
        self._ind_algorithms_by_terrain = {}
        self._pop_algorithms_by_terrain = {}

        # define training parameters
        self._batch_size = int(os.getenv('SNAKE_SAC_BATCH_SIZE', '32'))
        self._nmbr_ind_updates = int(os.getenv('SNAKE_IND_UPDATES', '25'))
        self._nmbr_pop_updates = int(os.getenv('SNAKE_POP_UPDATES', '8'))
        self._ind_replay_epochs_per_update = max(
            1.0,
            float(os.getenv('SNAKE_IND_REPLAY_EPOCHS_PER_UPDATE', '1.0')),
        )
        self._pop_replay_epochs_per_update = max(
            1.0,
            float(os.getenv('SNAKE_POP_REPLAY_EPOCHS_PER_UPDATE', '1.0')),
        )
        ind_policy_lr = float(os.getenv('SNAKE_IND_POLICY_LR', '1e-3'))
        ind_qf_lr = float(os.getenv('SNAKE_IND_QF_LR', '1e-3'))
        pop_policy_lr = float(os.getenv('SNAKE_POP_POLICY_LR', '1e-3'))
        pop_qf_lr = float(os.getenv('SNAKE_POP_QF_LR', '1e-3'))
        common_trainer_kwargs = dict(
            discount=0.99,
            reward_scale=1.0,
            optimizer_class=optim.Adam,
            soft_target_tau=.01,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
            alpha=0.1,
        )
        self._ind_sac_trainer_kwargs = dict(
            common_trainer_kwargs,
            policy_lr=ind_policy_lr,
            qf_lr=ind_qf_lr,
        )
        self._pop_sac_trainer_kwargs = dict(
            common_trainer_kwargs,
            policy_lr=pop_policy_lr,
            qf_lr=pop_qf_lr,
        )

        self._bind_active_terrain(self._active_terrain)
        self._ensure_trainers(self._active_terrain)

        self.last_ind_diagnostics = {}
        self.last_pop_diagnostics = {}

    def _new_network_group(self):
        return self._create_networks(env=self._env)

    def _ensure_terrain_networks(self, terrain_name):
        terrain_name = str(terrain_name)
        if self._terrain_model_mode == 'shared':
            terrain_name = 'shared'
        if terrain_name not in self._individual_networks_by_terrain:
            self._individual_networks_by_terrain[terrain_name] = self._new_network_group()
        if terrain_name not in self._population_networks_by_terrain:
            self._population_networks_by_terrain[terrain_name] = self._new_network_group()
        if terrain_name not in self._terrain_names:
            self._terrain_names.append(terrain_name)
            self._networks['terrain_names'] = list(self._terrain_names)
        return terrain_name

    def _bind_active_terrain(self, terrain_name):
        terrain_name = self._ensure_terrain_networks(terrain_name)
        self._active_terrain = terrain_name
        self._networks['individual'] = self._individual_networks_by_terrain[terrain_name]
        self._networks['population'] = self._population_networks_by_terrain[terrain_name]

        self._ind_qf1 = self._networks['individual']['qf1']
        self._ind_qf2 = self._networks['individual']['qf2']
        self._ind_qf1_target = self._networks['individual']['qf1_target']
        self._ind_qf2_target = self._networks['individual']['qf2_target']
        self._ind_policy = self._networks['individual']['policy']

        self._pop_qf1 = self._networks['population']['qf1']
        self._pop_qf2 = self._networks['population']['qf2']
        self._pop_qf1_target = self._networks['population']['qf1_target']
        self._pop_qf2_target = self._networks['population']['qf2_target']
        self._pop_policy = self._networks['population']['policy']

        self._ind_algorithm = self._ind_algorithms_by_terrain.get(terrain_name)
        self._pop_algorithm = self._pop_algorithms_by_terrain.get(terrain_name)
        return terrain_name

    def _build_individual_trainer_for_terrain(self, terrain_name):
        networks = self._individual_networks_by_terrain[terrain_name]
        return self._build_trainer(
            policy=networks['policy'],
            qf1=networks['qf1'],
            qf2=networks['qf2'],
            target_qf1=networks['qf1_target'],
            target_qf2=networks['qf2_target'],
            trainer_kwargs=self._ind_sac_trainer_kwargs,
        )

    def _build_population_trainer_for_terrain(self, terrain_name):
        networks = self._population_networks_by_terrain[terrain_name]
        return self._build_trainer(
            policy=networks['policy'],
            qf1=networks['qf1'],
            qf2=networks['qf2'],
            target_qf1=networks['qf1_target'],
            target_qf2=networks['qf2_target'],
            trainer_kwargs=self._pop_sac_trainer_kwargs,
        )

    def _ensure_trainers(self, terrain_name):
        terrain_name = self._ensure_terrain_networks(terrain_name)
        if terrain_name not in self._ind_algorithms_by_terrain:
            self._ind_algorithms_by_terrain[terrain_name] = self._build_individual_trainer_for_terrain(terrain_name)
        if terrain_name not in self._pop_algorithms_by_terrain:
            self._pop_algorithms_by_terrain[terrain_name] = self._build_population_trainer_for_terrain(terrain_name)
        self._bind_active_terrain(terrain_name)

    def set_active_terrain(self, terrain_name, reset_individual=False, reset_population=False):
        terrain_name = self._ensure_terrain_networks(terrain_name)
        if reset_individual:
            self._individual_networks_by_terrain[terrain_name] = self._new_network_group()
            self._networks['individual_by_terrain'] = self._individual_networks_by_terrain
            self._ind_algorithms_by_terrain[terrain_name] = self._build_individual_trainer_for_terrain(terrain_name)
        if reset_population:
            self._population_networks_by_terrain[terrain_name] = self._new_network_group()
            self._networks['population_by_terrain'] = self._population_networks_by_terrain
            self._pop_algorithms_by_terrain[terrain_name] = self._build_population_trainer_for_terrain(terrain_name)

        self._ensure_trainers(terrain_name)
        return terrain_name

    def get_terrain_networks(self, role, terrains=None):
        if role == 'individual':
            source = self._individual_networks_by_terrain
        elif role == 'population':
            source = self._population_networks_by_terrain
        else:
            raise ValueError(f"Unknown network role: {role}")

        if terrains is None:
            return dict(source)
        result = {}
        for terrain in terrains:
            terrain = self._ensure_terrain_networks(terrain)
            result[terrain] = source[terrain]
        return result

    def individual_terrains_available(self):
        return sorted(self._individual_networks_by_terrain.keys())

    def population_terrains_available(self):
        return sorted(self._population_networks_by_terrain.keys())
        

    def set_target_entropy(self, target_entropy):
        self._ind_sac_trainer_kwargs['target_entropy'] = float(target_entropy)
        self._pop_sac_trainer_kwargs['target_entropy'] = float(target_entropy)
        for algorithm in self._ind_algorithms_by_terrain.values():
            if hasattr(algorithm, 'set_target_entropy'):
                algorithm.set_target_entropy(target_entropy)
        for algorithm in self._pop_algorithms_by_terrain.values():
            if hasattr(algorithm, 'set_target_entropy'):
                algorithm.set_target_entropy(target_entropy)
    
    def episode_init(
            self,
            copy_population_to_individual=True,
            terrain_name=None,
            reset_individual=False,
    ):
           
        """Initialize the individual trainer for a fresh adaptation phase.

        When copy_population_to_individual is true, the individual networks
        start from the current population networks before adapting locally.
        """
        ptu.set_gpu_mode(False)
        if terrain_name is not None:
            self.set_active_terrain(terrain_name, reset_individual=reset_individual)
        elif reset_individual:
            self.set_active_terrain(self._active_terrain, reset_individual=True)
        else:
            self._ensure_trainers(self._active_terrain)

        if copy_population_to_individual:
            print('Copying population networks into individual networks')
            utils.copy_pop_to_ind(networks_pop=self._networks['population'], networks_ind=self._networks['individual'])
            self._ind_algorithms_by_terrain[self._active_terrain] = self._build_individual_trainer_for_terrain(self._active_terrain)
            self._bind_active_terrain(self._active_terrain)
        else:
            print(f'Keeping individual networks for terrain {self._active_terrain}')

    def _build_trainer(self, policy, qf1, qf2, target_qf1, target_qf2, trainer_kwargs):
        return SACTrainer(
            env=self._env,
            policy=policy,
            qf1=qf1,
            qf2=qf2,
            target_qf1=target_qf1,
            target_qf2=target_qf2,
            **trainer_kwargs,
        )
        
  
    def single_train_step(self, train_ind=True, train_pop=False, terrain_name=None):
        """
            single step in the training
        """
        ptu.set_gpu_mode(False)
        if terrain_name is not None:
            self.set_active_terrain(terrain_name)
        self.trainQ1losses = []
        self.trainQ2losses = []
        self.trainPolicylosses = []

        self.popQ1losses = [] 
        self.popQ2losses = [] 
        self.popPolicylosses = [] 
        self.last_ind_diagnostics = {}
        self.last_pop_diagnostics = {}

        print('IN TRAINING')
        if train_ind:
            self._replay.set_mode('species')
            ind_steps_can_sample = self._replay.num_steps_can_sample()
            ind_updates = min(
                self._nmbr_ind_updates,
                max(
                    1,
                    int(np.ceil(
                        self._ind_replay_epochs_per_update
                        * ind_steps_can_sample
                        / self._batch_size
                    )),
                )
            )
            print(f'individual SAC updates: {ind_updates} from {ind_steps_can_sample} replay steps')
            self._ind_algorithm.start_diagnostics_epoch()
            for i in range(ind_updates):
                #print('in ind for')
                batch = self._replay.random_batch(self._batch_size)
                self._ind_algorithm.train(batch)
                #print('trained')
            
            traindata = self._ind_algorithm.get_diagnostics()
            self.last_ind_diagnostics = dict(traindata)
            self.trainQ1losses.append(traindata['QF1 Loss'])
            self.trainQ2losses.append(traindata['QF2 Loss'])
            self.trainPolicylosses.append(traindata['Policy Loss'])
            self._ind_algorithm.end_epoch(1)

        if train_pop:
            self._replay.set_mode('population')
            pop_steps_can_sample = self._replay.num_steps_can_sample()
            pop_updates = min(
                self._nmbr_pop_updates,
                max(
                    1,
                    int(np.ceil(
                        self._pop_replay_epochs_per_update
                        * pop_steps_can_sample
                        / self._batch_size
                    )),
                )
            )
            print(f'population SAC updates: {pop_updates} from {pop_steps_can_sample} replay steps')
            self._pop_algorithm.start_diagnostics_epoch()
            for i in range(pop_updates):
                #print('in pop for')
                batch = self._replay.random_batch(self._batch_size)
                self._pop_algorithm.train(batch)

            traindataPop = self._pop_algorithm.get_diagnostics()
            self.last_pop_diagnostics = dict(traindataPop)
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
    def create_networks(env, terrain_names=None, terrain_model_mode=None):
        """ Creates all networks necessary for SAC.

        These networks have to be created before instantiating this class and
        used in the constructor.

        Returns:
            A dictonary which contains the networks.
        """
        ptu.set_gpu_mode(False)
        terrain_model_mode = str(
            terrain_model_mode or os.getenv('SNAKE_TERRAIN_MODEL_MODE', 'separate')
        ).strip().lower()
        if terrain_model_mode not in ('separate', 'shared'):
            raise ValueError(
                "SNAKE_TERRAIN_MODEL_MODE must be either 'separate' or 'shared'."
            )

        if terrain_names is None:
            terrain_names = [
                terrain.strip()
                for terrain in os.getenv('SNAKE_ACTIVE_TERRAINS', 'carpet,cardboard').split(',')
                if terrain.strip()
            ]
        terrain_names = list(terrain_names or ['default'])

        if terrain_model_mode == 'separate':
            individual_by_terrain = {
                terrain: SoftActorCriticCoadapt._create_networks(env=env)
                for terrain in terrain_names
            }
            population_by_terrain = {
                terrain: SoftActorCriticCoadapt._create_networks(env=env)
                for terrain in terrain_names
            }
            active_terrain = terrain_names[0]
            network_dict = {
                'terrain_model_mode': terrain_model_mode,
                'terrain_names': terrain_names,
                'individual_by_terrain': individual_by_terrain,
                'population_by_terrain': population_by_terrain,
                'individual': individual_by_terrain[active_terrain],
                'population': population_by_terrain[active_terrain],
            }
        else:
            network_dict = {
                'terrain_model_mode': terrain_model_mode,
                'terrain_names': ['shared'],
                'individual' : SoftActorCriticCoadapt._create_networks(env=env),
                'population' : SoftActorCriticCoadapt._create_networks(env=env),
            }
            network_dict['individual_by_terrain'] = {'shared': network_dict['individual']}
            network_dict['population_by_terrain'] = {'shared': network_dict['population']}
        
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
