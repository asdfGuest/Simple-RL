import torch as th
import torch.nn as nn
from torch.distributions.normal import Normal

import math
from typing import List, Union, Tuple, Type

from simple_rl.common.network import MlpNet



class PolicyNet(nn.Module) :

    def __init__(
            self,
            obs_dim:int,
            action_dim:int,
            net_arch:List[int]=[64,64],
            act_func:nn.Module=nn.Tanh,
            ortho_init:bool=True,
            init_std:float=1.0,
            clip_std:Tuple[float,float]=(None, None),
            grad_scale:float=1.0
            ) :
        super(PolicyNet, self).__init__()
        # build mlp
        self.net = MlpNet(
            input_dim=obs_dim,
            output_dim=action_dim,
            net_arch=net_arch,
            act_func=act_func
        )
        # orthogonal initialize
        if ortho_init :
            for layer in self.net.layers[:-1] :
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
            layer = self.net.layers[-1]
            nn.init.orthogonal_(layer.weight, gain=0.01)
            nn.init.constant_(layer.bias, 0.0)
        # std parameter
        self.grad_scale = grad_scale
        self.std_param = nn.Parameter(
            th.full((action_dim,), math.log(init_std)/self.grad_scale, dtype=th.float32),
            requires_grad=True
        )
        # clip_low <= e^(std_param * grad_scale) <= clip_high
        # log(clip_low) <= std_param * grad_scale <= log(clip_high)
        self.clip_low = math.log(clip_std[0])/self.grad_scale if clip_std[0] != None else None
        self.clip_up = math.log(clip_std[1])/self.grad_scale if clip_std[1] != None else None
    
    def forward(self, obs:th.Tensor, action:Union[Type[th.Tensor],None]=None, deterministic:bool=False) -> Tuple[Type[th.Tensor], Type[th.Tensor], Type[th.Tensor]] :
        # mu
        mu = self.net(obs)
        # std
        if self.clip_low != None or self.clip_up != None :
            self.std_param.data.clamp_(self.clip_low, self.clip_up)
        
        std = (self.std_param * self.grad_scale).exp()
        std = std.expand(mu.shape)
        # pdf
        pdf = Normal(mu, std)
        # sample action
        if action == None :
            action = pdf.sample() if not deterministic else mu
        # calculate log prob and entropy
        log_prob = pdf.log_prob(action).sum(dim=1,keepdim=True)
        entropy = pdf.entropy().sum(dim=1,keepdim=True)
        return action, log_prob, entropy



class ValueNet(nn.Module) :
    
    def __init__(
            self,
            obs_dim:int,
            net_arch:List[int]=[64,64],
            act_func:nn.Module=nn.Tanh,
            ortho_init:bool=True,
            ) :
        super(ValueNet, self).__init__()
        # build mlp
        self.net = MlpNet(
            input_dim=obs_dim,
            output_dim=1,
            net_arch=net_arch,
            act_func=act_func
        )
        # orthogonal initialize
        if ortho_init :
            for layer in self.net.layers[:-1] :
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
            layer = self.net.layers[-1]
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, obs:th.Tensor) -> Type[th.Tensor] :
        return self.net(obs)