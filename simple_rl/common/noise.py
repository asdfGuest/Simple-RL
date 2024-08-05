import torch as th
from typing import Tuple

import math



class GaussianNoise() :

    def __init__(self, shape:Tuple, mean:float=0.0, std:float=0.1) :
        self.pdf = th.distributions.Normal(
            th.full(shape, mean, dtype=th.float32),
            th.full(shape, std, dtype=th.float32)
        )
    
    def __call__(self) :
        return self.pdf.sample()



class OrnsteinUhlenbeckNoise() :
    def __init__(self, shape:Tuple, mu:float=0.0, theta:float=0.15, sigma:float=0.1, dt:float=0.001) :
        self.mu = mu
        self.theta = theta
        self.dt = dt
        self.sqrt_dt = math.sqrt(dt)

        self.normal = th.distributions.Normal(
            th.full(shape, 0.0, dtype=th.float32),
            th.full(shape, sigma, dtype=th.float32)
        )
        self.x = th.full(shape, mu, dtype=th.float32)

    def __call__(self) :
        self.x = (
            self.x +
            self.theta * (self.mu - self.x) * self.dt +
            self.normal.sample() * self.sqrt_dt
        )
        return self.x