import torch as th
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvSpec :
    n_env: int
    device: th.device
    n_obs: int
    n_action: int


class BaseEnv(ABC) :
    
    n_env: int
    '''
    number of parallel environments
    '''
    device: th.device
    '''
    '''
    n_obs: int
    '''
    dimension of observation observation vector
    '''
    n_action: int
    '''
    dimension of action vector
    '''

    @abstractmethod
    def reset(self) -> Tuple[th.Tensor, dict]:
        pass

    @abstractmethod
    def step(self, action:th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, dict]:
        pass

    @abstractmethod
    def close(self) :
        pass

    @property
    def spec(self) :
        return EnvSpec(
            n_env=self.n_env,
            device=self.device,
            n_obs=self.n_obs,
            n_action=self.n_action,
        )
