import torch as th
import time
from typing import Literal

from simple_rl.env import EnvSpec


class EpisodeTracker :
    def __init__(self, env_spec:EnvSpec) :
        self.done_sum = 0
        self.length_sum = 0
        self.return_sum = 0.
        self.length_track = th.ones(size=(env_spec.n_env,1), dtype=th.int32, device=env_spec.device)
        self.return_track = th.zeros(size=(env_spec.n_env,1), dtype=th.float32, device=env_spec.device)

    @property
    def mean_episode_length(self) :
        return self.length_sum / max(self.done_sum, 1)
    
    @property
    def mean_episode_return(self) :
        return self.return_sum / max(self.done_sum, 1)

    def update(self, reward:th.Tensor, done:th.Tensor) :
        self.done_sum += done.sum().item()
        self.length_sum += self.length_track[done].sum().item()
        self.return_sum += self.return_track[done].sum().item()

        self.length_track[done] = 0
        self.return_track[done] = 0.

        self.length_track += 1
        self.return_track += reward
    
    def clear(self) :
        self.done_sum = 0
        self.length_sum = 0
        self.return_sum = 0.


class StepTimer :
    def __init__(self, total_steps:int) :
        self.total_steps = total_steps
        self.taken_steps = 0
        self.start_time = time.time()
        self.update_time = self.start_time

    def spent_times(self, unit:Literal['h','m','s']='s') :
        t = time.time() - self.start_time
        if unit == 'h' :
            t /= 3600.
        elif unit == 'm' :
            t /= 60.
        return t

    def remaining_times(self, unit:Literal['h','m','s']='s') :
        t = (self.total_steps - self.taken_steps) * self.step_dt
        if unit == 'h' :
            t /= 3600.
        elif unit == 'm' :
            t /= 60.
        return t

    def update(self, delta_step:int=1) :
        self.taken_steps += delta_step
        self.last_update_time = self.update_time
        self.update_time = time.time()
        self.step_dt = (self.update_time - self.last_update_time) / delta_step
        self.fps = 1. / self.step_dt
