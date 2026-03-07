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

        design_dim = len(design)
        state_dim = initial_state.shape[1]
        design_idx = SnakeEnv.get_design_dimensions()
        valid_design_idx = [int(i) for i in design_idx if 0 <= int(i) < state_dim]
        if len(valid_design_idx) != design_dim:
            valid_design_idx = list(range(state_dim - design_dim, state_dim))

        lower_bounds = np.array([l for l, _ in SnakeEnv.design_parameter_bounds], dtype=np.float32)
        upper_bounds = np.array([u for _, u in SnakeEnv.design_parameter_bounds], dtype=np.float32)

        def _discretize_design(x):
            return np.clip(np.rint(x), lower_bounds, upper_bounds).astype(np.float32)

        def _inject_design(observation_batch, x_design):
            updated = observation_batch.copy()
            updated[:, valid_design_idx] = x_design
            return updated

        def _terrain_state_batches():
            """Create per-terrain state batches for robust aggregation.

            Replay currently does not carry explicit terrain ids in start-mode
            sampling. We therefore draw one independent start batch per terrain
            to approximate terrain-conditioned uncertainty in the objective.
            """
            terrain_batches = []
            for _ in SnakeEnv.terrains:
                try:
                    batch = self._replay.random_batch(self._state_batch_size)['observations']
                except Exception:
                    batch = initial_state
                terrain_batches.append(batch)
            return terrain_batches

        terrain_state_batches = _terrain_state_batches()

        def f_qval(x_input, **kwargs):  # function to optimize
            shape = x_input.shape
            cost = np.zeros((shape[0],))

            with torch.no_grad():
                for i in range(shape[0]):
                    x_discrete = _discretize_design(x_input[i])
                    terrain_returns = []
                    for terrain_state_batch in terrain_state_batches:
                        state_batch = _inject_design(terrain_state_batch, x_discrete)

                        network_input = torch.from_numpy(state_batch).to(device=ptu.device, dtype=torch.float32)
                        action_dist = policy_network(network_input)
                        if isinstance(action_dist, tuple):
                            action = action_dist[0]
                        else:
                            action = action_dist.sample()
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

        optimizer = ps.single.GlobalBestPSO(n_particles=700, dimensions=len(design), bounds=bounds, options=options)
        
        # Perform optimization
        cost, new_design = optimizer.optimize(f_qval, print_step=100, iters=5, verbose=3) #, n_processes=2) # iter was 250
        print('OPTIMIZED')
        return cost, new_design
