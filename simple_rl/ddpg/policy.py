import torch as th
import torch.nn as nn

from typing import List

from simple_rl.common.network import MlpNet


class PiNet(nn.Module) :

    def __init__(
            self,
            obs_dim:int,
            act_dim:int,
            net_arch:List[int]=[256,256],
            act_func:nn.Module=nn.SiLU
            ) :
        super(PiNet, self).__init__()
        # build mlp
        self.net = MlpNet(
            input_dim=obs_dim,
            output_dim=act_dim,
            net_arch=net_arch,
            act_func=act_func
        )
        self.tanh = nn.Tanh()
    
    def forward(self, obs:th.Tensor) -> th.Tensor :
        return self.tanh(self.net(obs))



class QNet(nn.Module) :
    
    def __init__(
            self,
            obs_dim:int,
            act_dim:int,
            net_arch:List[int]=[256,256],
            act_func:nn.Module=nn.SiLU
            ) :
        super(QNet, self).__init__()
        # build mlp
        self.net = MlpNet(
            input_dim=obs_dim+act_dim,
            output_dim=1,
            net_arch=net_arch,
            act_func=act_func
        )
    
    def forward(self, obs:th.Tensor, action:th.Tensor) -> th.Tensor :
        return self.net(th.cat((obs,action),dim=1))