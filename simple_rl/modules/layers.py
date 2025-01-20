import torch as th

from typing import List, Type


class MLP(th.nn.Module) :
    def __init__(self, net_arch:List[int], activ_fn:Type[th.nn.Module], activ_output:bool=False) :
        th.nn.Module.__init__(self)

        self.net = th.nn.Sequential()
        for k in range(len(net_arch) - 1) :
            self.net.append(th.nn.Linear(net_arch[k], net_arch[k+1]))
            self.net.append(th.nn.Identity() if (k == len(net_arch) - 2 and not activ_output) else activ_fn())

    def forward(self, x:th.Tensor) :
        return self.net(x)
