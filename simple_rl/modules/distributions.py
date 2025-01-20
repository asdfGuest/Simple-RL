import torch as th
import math


class DiagonalGaussian :
    def __init__(self, mean:th.Tensor, std:th.Tensor) :
        self.mean, self.std = th.broadcast_tensors(mean, std)
    
    def sample(self) :
        return th.normal(self.mean, self.std)
    
    def rsample(self) :
        return self.mean + self.std * th.randn_like(self.mean)
    
    def log_prob(self, x:th.Tensor) :
        return th.sum(
            - ((self.mean - x) ** 2) / (2 * (self.std ** 2))
            - th.log(self.std)
            - 0.5 * math.log(2 * math.pi)
        , dim=-1, keepdim=True)

    def entropy(self) :
        return th.sum(0.5 + 0.5 * math.log(2 * math.pi) + th.log(self.std), dim=-1, keepdim=True)
    
    @staticmethod
    def kld(p:'DiagonalGaussian', q:'DiagonalGaussian') :
        return th.sum(
            + (th.log(q.std) - th.log(p.std))
            + (p.std ** 2 + (p.mean - q.mean) ** 2) / (2 * (q.std ** 2))
            - 0.5
        , dim=-1, keepdim=True)
