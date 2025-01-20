import torch as th
import numpy as np
from typing import Literal


def to_python_float(x) :
    if isinstance(x, float):
        return x
    elif isinstance(x, int) :
        return float(x)
    elif isinstance(x, th.Tensor):
        if x.numel() == 1:
            return float(x.item())
    elif isinstance(x, np.ndarray) :
        if x.size == 1 :
            return float(x.item())
    elif isinstance(x, (np.float16, np.float32, np.float64)) :
        return float(x)
    return x


def time_to_str(value:float, unit:Literal['h','m','s']='s') :
    if unit == 'h' :
        value *= 3600.
    elif unit == 'm' :
        value *= 60.
    
    h = value // 3600.
    value -= 3600. * h
    m = value // 60.
    value -= 60. * m
    s = value

    return '%01.0f:%02.0f:%02.0f'%(h, m, s)
