import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
import pyswarms as ps
from design_optimization import Design_Optimization
from snakeenv_thread_coadapt import SnakeEnv
#from snakeenv_thread_coadapt import SnakeEnv

class PSO_batch(Design_Optimization):

    def __init__(self, replay, env):
        self._replay = replay
        self._env = env

        self._state_batch_size = 32

    def optimize_design(self, design, q_network, policy_network):
        self._replay.set_mode('start')

        try:
            initial_state = self._replay.random_batch(self._state_batch_size)['observations']
        except ValueError:
            # Fallback: if start-state buffer is empty, bootstrap from species data.
            self._replay.set_mode('species')
            initial_state = self._replay.random_batch(self._state_batch_size)['observations']
            self._replay.set_mode('start')

        design_dim = len(SnakeEnv.design_parameter_bounds)
        state_dim = initial_state.shape[1]
        design_feature_dim = len(SnakeEnv.config_numpy)
        design_idx = SnakeEnv.get_design_dimensions()
        valid_design_idx = [int(i) for i in design_idx if 0 <= int(i) < state_dim]
        if len(valid_design_idx) != design_feature_dim:
            valid_design_idx = list(range(state_dim - design_feature_dim, state_dim))

        lower_bounds = np.array([l for l, _ in SnakeEnv.design_parameter_bounds], dtype=np.float32)
        upper_bounds = np.array([u for _, u in SnakeEnv.design_parameter_bounds], dtype=np.float32)
        design_options = np.asarray(SnakeEnv.design_parameter_options, dtype=np.float32)

        def _discretize_design(x):
            values = np.asarray(x, dtype=np.float32).reshape(-1)[:design_dim]
            if len(values) < design_dim:
                values = np.pad(
                    values,
                    (0, design_dim - len(values)),
                    constant_values=design_options[0],
                )
            values = np.clip(values, lower_bounds, upper_bounds)
            nearest_idx = np.argmin(np.abs(values[:, None] - design_options[None, :]), axis=1)
            return design_options[nearest_idx].astype(np.float32)

        def _inject_design(observation_batch, x_design):
            updated = observation_batch.copy()
            encoded_design = SnakeEnv.encode_design_vector(x_design)
            updated[:, valid_design_idx] = encoded_design
            return updated

        def _terrain_state_batches():
            """Create terrain-conditioned start-state batches for robust scoring."""
            terrain_batches = []
            for terrain_id, terrain_name in enumerate(SnakeEnv.terrains):
                try:
                    if hasattr(self._replay, 'random_start_batch_for_terrain'):
                        sampled = self._replay.random_start_batch_for_terrain(
                            self._state_batch_size,
                            terrain_id,
                        )
                    else:
                        sampled = self._replay.random_batch(self._state_batch_size)
                    batch_obs = sampled['observations']
                except Exception:
                    batch_obs = initial_state

                terrain_batches.append((terrain_name, batch_obs))
            return terrain_batches

        terrain_state_batches = _terrain_state_batches()

        def _deterministic_actions(observation_tensor):
            deterministic_forward = False
            try:
                policy_output = policy_network(observation_tensor, deterministic=True)
                deterministic_forward = True
            except TypeError:
                policy_output = policy_network(observation_tensor)

            if isinstance(policy_output, (tuple, list)):
                if deterministic_forward and len(policy_output) > 0 and torch.is_tensor(policy_output[0]):
                    return policy_output[0]
                if len(policy_output) > 1 and torch.is_tensor(policy_output[1]):
                    return torch.tanh(policy_output[1])
                if len(policy_output) > 0 and torch.is_tensor(policy_output[0]):
                    return policy_output[0]

            normal_mean = getattr(policy_output, 'normal_mean', None)
            if torch.is_tensor(normal_mean):
                return torch.tanh(normal_mean)

            mean_action = getattr(policy_output, 'mean', None)
            if torch.is_tensor(mean_action):
                if torch.max(torch.abs(mean_action)).item() > 1.0 + 1e-6:
                    mean_action = torch.tanh(mean_action)
                return mean_action

            if hasattr(policy_network, 'get_action'):
                actions = []
                for obs in observation_tensor:
                    obs_np = obs.detach().cpu().numpy()
                    try:
                        action, _ = policy_network.get_action(obs_np, deterministic=True)
                    except TypeError:
                        action, _ = policy_network.get_action(obs_np)
                    actions.append(action)
                return torch.as_tensor(np.asarray(actions), device=observation_tensor.device, dtype=torch.float32)

            raise TypeError(f'Unsupported policy output type for deterministic PSO actions: {type(policy_output)}')


        def f_qval(x_input, **kwargs):  # function to optimize
            shape = x_input.shape
            cost = np.zeros((shape[0],))

            with torch.no_grad():
                for i in range(shape[0]):
                    x_discrete = _discretize_design(x_input[i])
                    terrain_returns = []
                    for _terrain_name, terrain_state_batch in terrain_state_batches:
                        state_batch = _inject_design(terrain_state_batch, x_discrete)

                        network_input = torch.from_numpy(state_batch).to(device=ptu.device, dtype=torch.float32)
                        action = _deterministic_actions(network_input)
                        output = q_network(network_input, action)
                        # J_t: predicted return for terrain t under candidate design.
                        terrain_returns.append(float(output.mean().item()))

                    terrain_returns = np.array(terrain_returns, dtype=np.float32)
                    # Maximize mean(J_t) - lambda * std(J_t) for uniform terrain
                    # performance; PSO minimizes, so negate the objective.
                    robustness_lambda = 0.5
                    robust_objective = terrain_returns.mean() - robustness_lambda * terrain_returns.std()
                    cost[i] = float(-robust_objective)

            return cost

        bounds = (lower_bounds, upper_bounds)

       
        # c1 = cognitive parameter
        # c2 = social parameter
        # w = inertia parameter
        # https://pyswarms.readthedocs.io/en/latest/api/pyswarms.single.html
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

        optimizer = ps.single.GlobalBestPSO(n_particles=700, dimensions=design_dim, bounds=bounds, options=options)
        
        # Perform optimization
        cost, new_design = optimizer.optimize(f_qval, print_step=100, iters=5, verbose=3) #, n_processes=2) # iter was 250
        new_design = _discretize_design(new_design)
        print('OPTIMIZED')
        return cost, new_design

