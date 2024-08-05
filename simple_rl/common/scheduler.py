from typing import Callable, Union

Scheduler = Callable[[float], float]

def get_scalar_scheduler(c:float) -> Scheduler :
    def f(x:float) :
        return c
    return f

def to_scheduler(x:Union[float,Scheduler,None]) -> Union[Scheduler,None] :
    return get_scalar_scheduler(x) if isinstance(x, float) else x

def get_linear_scheduler(start:float, end:float, end_frac:float=1.0) -> Scheduler :
    def f(x:float) :
        x = min(x / end_frac, 1.0)
        return start * (1.0 - x) + end * (x)
    return f