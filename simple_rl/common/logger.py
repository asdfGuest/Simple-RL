
from typing import Union, Dict, List, Tuple

class Logger() :
    def __init__(self) :
        self.x:Dict[List[Union[int,float]]] = dict()
        self.y:Dict[List[Union[int,float]]] = dict()

    @property
    def keys(self) :
        return self.x.keys()
    
    def register_key(self, key:str) :
        self.x[key] = []
        self.y[key] = []
    
    def add(self, key:str, x:Union[int,float], y:Union[int,float]) :
        if key not in self.x :
            self.register_key(key)
        self.x[key].append(x)
        self.y[key].append(y)
    
    def raw(self, key:str) -> Tuple[List, List] :
        return self.x[key], self.y[key]

    def get(self, key:str, window:int=1) -> Tuple[List, List] :
        mean = lambda a : sum(a) / len(a)

        x = self.x[key][window-1:]
        y = [mean(self.y[key][k:k+window]) for k in range(len(x))]
        
        return x, y