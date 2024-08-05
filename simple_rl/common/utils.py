import torch

from typing import Union


def floatTensor(x) :
    return torch.tensor(x,dtype=torch.float32)


def boolTensor(x) :
    return torch.tensor(x,dtype=torch.bool)


def toFloat(x:Union[int,float,None]) :
    return float(x) if isinstance(x,int) else x