from collections import OrderedDict
import inspect

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer


class SACTrainer(TorchTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            alpha=1.0,
            
    ):
        self.optimizer_weight_decay = 1E-4
        super().__init__()
        self.env = env
        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -np.prod(self.env.action_space.shape).item()  # heuristic value from Tuomas
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.log_alpha.data.fill_(float(np.log(max(alpha, 1e-6))))
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self._alpha = alpha

        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(), # no weight decay for policy
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            weight_decay=self.optimizer_weight_decay,
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            weight_decay=self.optimizer_weight_decay,
            lr=qf_lr,
        )

        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self._policy_forward_signature = inspect.signature(self.policy.forward)
        self._reset_diagnostic_history()

    def _reset_diagnostic_history(self):
        self._diagnostic_history = OrderedDict(
            [
                ('QF1 Loss', []),
                ('QF2 Loss', []),
                ('Policy Loss', []),
                ('Alpha', []),
                ('Alpha Loss', []),
            ]
        )

    def _record_diagnostic(self, key, value):
        if key not in self._diagnostic_history:
            self._diagnostic_history[key] = []
        self._diagnostic_history[key].append(float(value))

    def start_diagnostics_epoch(self):
        self._reset_diagnostic_history()

    def _refresh_eval_statistics(self, q1_pred, q2_pred, q_target, log_pi, policy_mean, policy_log_std):
        scalar_stats = OrderedDict()
        for key, values in self._diagnostic_history.items():
            if values:
                scalar_stats[key] = float(np.mean(values))

        self.eval_statistics = OrderedDict()
        self.eval_statistics.update(scalar_stats)
        self.eval_statistics['Train Updates'] = len(self._diagnostic_history['QF1 Loss'])
        self.eval_statistics.update(create_stats_ordered_dict(
            'Q1 Predictions',
            ptu.get_numpy(q1_pred),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Q2 Predictions',
            ptu.get_numpy(q2_pred),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Q Targets',
            ptu.get_numpy(q_target),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Log Pis',
            ptu.get_numpy(log_pi),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Policy mu',
            ptu.get_numpy(policy_mean),
        ))
        self.eval_statistics.update(create_stats_ordered_dict(
            'Policy log std',
            ptu.get_numpy(policy_log_std),
        ))
        if not self.use_automatic_entropy_tuning:
            self.eval_statistics['Alpha'] = float(self._alpha)

    def _policy_forward(self, obs):
        policy_kwargs = {}
        if 'reparameterize' in self._policy_forward_signature.parameters:
            policy_kwargs['reparameterize'] = True
        elif 'reparametrize' in self._policy_forward_signature.parameters:
            # Compatibility with policies that use the misspelled keyword.
            policy_kwargs['reparametrize'] = True
        if 'return_log_prob' in self._policy_forward_signature.parameters:
            policy_kwargs['return_log_prob'] = True
        policy_output = self.policy(obs, **policy_kwargs)

        if isinstance(policy_output, (tuple, list)):
            return policy_output

        # Some rlkit variants return a distribution object (e.g. TanhNormal)
        # instead of an unpackable tuple. In that case, sample an action and
        # derive log-prob / summary stats manually.
        dist = policy_output
        pre_tanh_value = None

        if hasattr(dist, 'rsample'):
            try:
                actions, pre_tanh_value = dist.rsample(return_pretanh_value=True)
            except TypeError:
                actions = dist.rsample()
        elif hasattr(dist, 'sample'):
            try:
                actions, pre_tanh_value = dist.sample(return_pretanh_value=True)
            except TypeError:
                actions = dist.sample()
        else:
            raise TypeError(f"Unsupported policy output type: {type(dist)}")

        log_pi = None
        if hasattr(dist, 'log_prob'):
            try:
                if pre_tanh_value is not None:
                    log_pi = dist.log_prob(actions, pre_tanh_value=pre_tanh_value)
                else:
                    log_pi = dist.log_prob(actions)
            except TypeError:
                log_pi = dist.log_prob(actions)

            if log_pi.dim() == 1:
                log_pi = log_pi.unsqueeze(-1)
            elif log_pi.dim() > 1 and log_pi.shape[-1] != 1:
                log_pi = log_pi.sum(dim=-1, keepdim=True)

        if log_pi is None:
            log_pi = torch.zeros_like(actions[..., :1])

        policy_mean = getattr(dist, 'normal_mean', None)
        if policy_mean is None:
            policy_mean = getattr(dist, 'mean', None)
        if policy_mean is None:
            policy_mean = actions

        policy_log_std = getattr(dist, 'normal_log_std', None)
        if policy_log_std is None:
            policy_log_std = getattr(dist, 'log_std', None)
        if policy_log_std is None:
            std = getattr(dist, 'normal_std', None)
            if std is None:
                std = getattr(dist, 'std', None)
            if std is not None and torch.is_tensor(std):
                policy_log_std = torch.log(std.clamp(min=1e-6))
            else:
                policy_log_std = torch.zeros_like(actions)

        return actions, policy_mean, policy_log_std, log_pi


    def train_from_torch(self, batch):
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        """
        Policy and Alpha Loss
        """

        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self._policy_forward(obs)
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self._alpha

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )
        policy_loss = (alpha*log_pi - q_new_actions).mean()
        policy_loss_value = float(np.mean(ptu.get_numpy(policy_loss.detach())))

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, _, _, new_log_pi, *_ = self._policy_forward(next_obs)
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values
        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        """
        Soft Updates
        """
        if self._n_train_steps_total % self.target_update_period == 0:
            ptu.soft_update_from_to(
                self.qf1, self.target_qf1, self.soft_target_tau
            )
            ptu.soft_update_from_to(
                self.qf2, self.target_qf2, self.soft_target_tau
            )

        """
        Save some statistics for eval
        """
        self._need_to_update_eval_statistics = False
        self._record_diagnostic('QF1 Loss', np.mean(ptu.get_numpy(qf1_loss.detach())))
        self._record_diagnostic('QF2 Loss', np.mean(ptu.get_numpy(qf2_loss.detach())))
        self._record_diagnostic('Policy Loss', policy_loss_value)
        if self.use_automatic_entropy_tuning:
            self._record_diagnostic('Alpha', alpha.item())
            self._record_diagnostic('Alpha Loss', alpha_loss.item())
        self._refresh_eval_statistics(
            q1_pred=q1_pred,
            q2_pred=q2_pred,
            q_target=q_target,
            log_pi=log_pi,
            policy_mean=policy_mean,
            policy_log_std=policy_log_std,
        )
        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._reset_diagnostic_history()
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        return [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )
