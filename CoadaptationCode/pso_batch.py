import os
import numpy as np
import torch
import rlkit.torch.pytorch_util as ptu
import pyswarms as ps
from design_optimization import Design_Optimization
from snakeenv_thread_coadapt import SnakeEnv


class PSO_batch(Design_Optimization):

    def __init__(self, replay, env):
        self._replay = replay
        self._env = env

        self._state_batch_size = 32

    def optimize_design(
            self,
            design,
            q_network,
            policy_network,
            active_terrains=None,
            design_mode='heterogeneous',
    ):
        previous_replay_mode = getattr(self._replay, '_mode', 'species')
        self._replay.set_mode('start')

        active_terrains = list(active_terrains or SnakeEnv.terrains)
        invalid_terrains = [terrain for terrain in active_terrains if terrain not in SnakeEnv.terrains]
        if invalid_terrains:
            raise ValueError(
                f"Unknown active terrain(s) for PSO: {invalid_terrains}. "
                f"Use one of {SnakeEnv.terrains}."
            )

        try:
            initial_state = self._replay.random_batch(self._state_batch_size)['observations']
        except ValueError:
            # Fallback: if start-state buffer is empty, bootstrap from species data.
            self._replay.set_mode('species')
            initial_state = self._replay.random_batch(self._state_batch_size)['observations']
            self._replay.set_mode('start')

        state_dim = initial_state.shape[1]
        design_feature_dim = len(SnakeEnv.config_numpy)
        design_idx = SnakeEnv.get_design_dimensions()
        valid_design_idx = [int(i) for i in design_idx if 0 <= int(i) < state_dim]
        if len(valid_design_idx) != design_feature_dim:
            valid_design_idx = list(range(state_dim - design_feature_dim, state_dim))

        opt_bounds = SnakeEnv.get_optimization_bounds(design_mode)
        lower_bounds = np.asarray([lower for lower, _upper in opt_bounds], dtype=np.float32)
        upper_bounds = np.asarray([upper for _lower, upper in opt_bounds], dtype=np.float32)
        dimensions = len(opt_bounds)
        robustness_lambda = float(os.getenv('SNAKE_PSO_ROBUSTNESS_LAMBDA', '0.5'))
        default_min_heterogeneity_delta = (
            '0.1' if str(design_mode).strip().lower() == 'heterogeneous' else '0.0'
        )
        min_heterogeneity_delta = max(
            0.0,
            float(os.getenv('SNAKE_MIN_HETEROGENEITY_DELTA', default_min_heterogeneity_delta)),
        )

        def _candidate_to_full_design(candidate):
            return np.asarray(
                SnakeEnv.expand_optimization_design(candidate, design_mode),
                dtype=np.float32,
            )

        def _candidate_vector(full_design):
            full_design = SnakeEnv._coerce_design_vector(full_design)
            if str(design_mode).strip().lower() == 'homogeneous':
                return np.asarray([full_design[0], full_design[1]], dtype=np.float32)
            return np.asarray(full_design, dtype=np.float32)

        def _heterogeneity_distance(full_design):
            full_design = SnakeEnv._coerce_design_vector(full_design)
            width_span = max(
                SnakeEnv.design_parameter_bounds[0][1] - SnakeEnv.design_parameter_bounds[0][0],
                1e-6,
            )
            attack_angle_span = max(
                SnakeEnv.design_parameter_bounds[1][1] - SnakeEnv.design_parameter_bounds[1][0],
                1e-6,
            )
            width_delta = abs(full_design[0] - full_design[2]) / width_span
            attack_angle_delta = abs(full_design[1] - full_design[3]) / attack_angle_span
            return float(np.sqrt(width_delta ** 2 + attack_angle_delta ** 2))

        def _violates_design_constraints(full_design):
            if str(design_mode).strip().lower() != 'heterogeneous':
                return False
            if min_heterogeneity_delta <= 0.0:
                return False
            return _heterogeneity_distance(full_design) < min_heterogeneity_delta

        def _inject_design(observation_batch, full_design):
            updated = observation_batch.copy()
            encoded_design = SnakeEnv.encode_design_vector(full_design)
            updated[:, valid_design_idx] = encoded_design
            return updated

        def _terrain_state_batches():
            """Create terrain-conditioned start-state batches for robust scoring."""
            terrain_batches = []
            for terrain_name in active_terrains:
                terrain_id = SnakeEnv.get_terrain_id(terrain_name)
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

        def f_qval(x_input, **kwargs):
            shape = x_input.shape
            cost = np.zeros((shape[0],), dtype=np.float32)

            with torch.no_grad():
                for i in range(shape[0]):
                    full_design = _candidate_to_full_design(x_input[i])
                    if _violates_design_constraints(full_design):
                        cost[i] = 1e6
                        continue

                    terrain_returns = []
                    for _terrain_name, terrain_state_batch in terrain_state_batches:
                        state_batch = _inject_design(terrain_state_batch, full_design)

                        network_input = torch.from_numpy(state_batch).to(device=ptu.device, dtype=torch.float32)
                        action = _deterministic_actions(network_input)
                        output = q_network(network_input, action)
                        terrain_returns.append(float(output.mean().item()))

                    terrain_returns = np.asarray(terrain_returns, dtype=np.float32)
                    robust_objective = terrain_returns.mean() - robustness_lambda * terrain_returns.std()
                    cost[i] = float(-robust_objective)

            return cost

        bounds = (lower_bounds, upper_bounds)

        # c1 = cognitive parameter, c2 = social parameter, w = inertia parameter.
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

        n_particles = max(1, int(os.getenv('SNAKE_PSO_PARTICLES', '700')))
        pso_iters = max(1, int(os.getenv('SNAKE_PSO_ITERS', '5')))
        init_pos = np.random.uniform(
            low=lower_bounds,
            high=upper_bounds,
            size=(n_particles, dimensions),
        ).astype(np.float32)
        if design is not None:
            init_pos[0] = np.clip(_candidate_vector(design), lower_bounds, upper_bounds)

        optimizer = ps.single.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=dimensions,
            bounds=bounds,
            options=options,
            init_pos=init_pos,
        )

        cost, best_candidate = optimizer.optimize(f_qval, print_step=100, iters=pso_iters, verbose=3)
        best_design = _candidate_to_full_design(best_candidate)
        print('OPTIMIZED')
        print('Best scale design:', best_design)
        print('Best scale design cost:', cost)
        self._replay.set_mode(previous_replay_mode)
        return cost, best_design
