import torch as th
import math

from abc import abstractmethod
from typing import List, Type

from simple_rl.modules.layers import MLP
from simple_rl.modules.distributions import DiagonalGaussian


class BasePolicy(th.nn.Module) :
    def __init__(self) :
        super().__init__()

    @abstractmethod
    def compute(self, obs:th.Tensor) -> DiagonalGaussian:
        pass

    @abstractmethod
    def get_pdf_cls(self) -> Type[DiagonalGaussian] :
        pass


class BaseValue(th.nn.Module) :
    def __init__(self) :
        super().__init__()

    @abstractmethod
    def compute(self, obs:th.Tensor) -> th.Tensor:
        pass


class MlpPolicy(BasePolicy) :
    def __init__(
            self,
            n_obs:int,
            n_action:int,
            init_std:float,
            net_arch:List[int],
            activ_fn:Type[th.nn.Module]
        ):
        super().__init__()

        self.n_obs = n_obs
        self.n_action = n_action

        net_arch = [n_obs] + net_arch + [n_action]
        self.mean = MLP(net_arch, activ_fn)
        self.logstd = th.nn.Parameter(th.full(size=(1,self.n_action), fill_value=math.log(init_std)))

    def compute(self, obs:th.Tensor) :
        mean = self.mean(obs)
        std = th.exp(self.logstd).expand_as(mean)
        return DiagonalGaussian(mean, std)
    
    def get_pdf_cls(self):
        return DiagonalGaussian


class MlpValue(BaseValue) :
    def __init__(
            self,
            n_obs:int,
            net_arch:List[int],
            activ_fn:Type[th.nn.Module]
        ):
        super().__init__()

        net_arch = [n_obs] + net_arch + [1]
        self.mlp = MLP(net_arch, activ_fn)
    
    def compute(self, obs:th.Tensor):
        return self.mlp(obs)
