import torch as th
from typing import List

from simple_rl.algorithms.ppo.config import PPOCfg


class PPOBuffer :

    # preliminary
    obs: List[th.Tensor]
    action: List[th.Tensor]
    action_mean: List[th.Tensor]
    action_std: List[th.Tensor]
    action_logprob: List[th.Tensor]
    reward: List[th.Tensor]
    done: List[th.Tensor]
    value: List[th.Tensor]
    # computed
    advantage: List[th.Tensor]
    returns: List[th.Tensor]

    def __init__(self, cfg:PPOCfg) :
        self.cfg = cfg
        self.clear()

    def clear(self) :
        self.obs = []
        self.action = []
        self.action_mean = []
        self.action_std = []
        self.action_logprob = []
        self.reward = []
        self.done = []
        self.value = []
        self.advantage = []
        self.returns = []

    def push(
            self,
            obs:th.Tensor,
            action:th.Tensor,
            action_mean:th.Tensor,
            action_std:th.Tensor,
            action_logprob:th.Tensor,
            reward:th.Tensor,
            done:th.Tensor,
            value:th.Tensor
        ) :
        self.obs.append(obs.detach())
        self.action.append(action.detach())
        self.action_mean.append(action_mean.detach())
        self.action_std.append(action_std.detach())
        self.action_logprob.append(action_logprob.detach())
        self.reward.append(reward.detach())
        self.done.append(done.detach())
        self.value.append(value.detach())
        self.advantage.append(None)
        self.returns.append(None)

    def compute_gae(self, next_value:th.Tensor) :
        next_advantage = th.zeros_like(next_value)
        
        for idx in reversed(range(self.size)):
            not_done = (~self.done[idx]).to(th.float32)
            delta = self.reward[idx] + not_done * (self.cfg.gamma * next_value) - self.value[idx]
            self.advantage[idx] = delta + not_done * (self.cfg.gamma * self.cfg.gae_lambda) * next_advantage
            self.returns[idx] = self.advantage[idx] + self.value[idx]

            next_value = self.value[idx]
            next_advantage = self.advantage[idx]
    
    def get_minibatch_generator(self) :
        batch_list = [
            th.cat(self.obs),
            th.cat(self.action),
            th.cat(self.action_mean),
            th.cat(self.action_std),
            th.cat(self.action_logprob),
            th.cat(self.value),
            th.cat(self.advantage),
            th.cat(self.returns),
        ]

        batch_size = batch_list[0].shape[0]
        minibatch_size = batch_size // self.cfg.n_minibatch # drop last

        for _ in range(self.cfg.n_epoch) :
            rand_indices = th.randperm(batch_size, device=batch_list[0].device)
            batch_list = [batch[rand_indices] for batch in batch_list]
            
            for minibatch_idx in range(self.cfg.n_minibatch) :
                low = minibatch_size * minibatch_idx
                high = minibatch_size * (minibatch_idx + 1)
                yield [batch[low:high] for batch in batch_list]

    @property
    def size(self) :
        return len(self.obs)
