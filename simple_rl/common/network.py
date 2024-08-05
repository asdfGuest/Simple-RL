import torch as th
import torch.nn as nn

from typing import List



class MlpNet(nn.Module) :

    def __init__(
            self,
            input_dim:int,
            output_dim:int,
            net_arch:List[int],
            act_func:nn.Module,
            last_layer_active=False
        ) :
        super(MlpNet, self).__init__()
        net_arch = [input_dim] + net_arch + [output_dim]
        # build mlp
        self.layers = nn.ModuleList()
        self.activs = nn.ModuleList()
        
        layer_num = len(net_arch) - 1
        for k in range(layer_num) :
            self.layers.append(nn.Linear(net_arch[k], net_arch[k+1]))
            self.activs.append(act_func() if k<layer_num-1 or last_layer_active else nn.Identity())
    
    def forward(self, x:th.Tensor) :
        for layer, activ in zip(self.layers, self.activs) :
            x = layer(x)
            x = activ(x)
        return x