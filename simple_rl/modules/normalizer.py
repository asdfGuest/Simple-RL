import torch as th
from typing import Tuple


class Normalizer(th.nn.Module) :
    def __init__(self, shape:Tuple[int,...], clip:float=10.0, eps:float=1e-4) :
        super().__init__()
        self.shape = shape
        self.clip = clip
        self.eps = eps

        self.register_buffer('mean', th.zeros((1,) + shape, dtype=th.float32))
        self.register_buffer('var', th.ones((1,) + shape, dtype=th.float32))
        self.register_buffer('std', th.ones((1,) + shape, dtype=th.float32))
        self.register_buffer('cnt', th.zeros((1,), dtype=th.int64))

        self.mean:th.Tensor = self.mean
        self.var:th.Tensor = self.var
        self.std:th.Tensor = self.std
        self.cnt:th.Tensor = self.cnt
    
    def _update(self, x:th.Tensor) :
        x_cnt = x.shape[0]
        x_mean = th.mean(x, dim=0, keepdim=True)
        x_var = th.var(x, dim=0, keepdim=True, unbiased=False)

        new_cnt = self.cnt + x_cnt
        ratio = x_cnt / new_cnt
        delta_mean = x_mean - self.mean

        new_mean = self.mean + ratio * delta_mean
        new_var = self.var + ratio * (x_var - self.var + delta_mean * (x_mean - new_mean))

        self.cnt.copy_(new_cnt)
        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.std.copy_(th.sqrt(self.var+self.eps))
    
    def normalize(self, x:th.Tensor, update:bool=False) :
        if update:
            self._update(x)
        return th.clip((x - self.mean) / self.std, -self.clip, self.clip)
    
    def denormalize(self, x:th.Tensor) :
        return x * self.std + self.mean
    
    def forward(self, x:th.Tensor, update:bool=False) :
        return self.normalize(x, update)
