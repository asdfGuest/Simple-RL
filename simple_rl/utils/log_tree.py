from typing import List, Tuple, Dict, Any, Union


class DictTree :
    def __init__(self, is_leaf:bool, value=None, src:Dict[str,'DictTree']=None) :
        self.is_leaf = is_leaf
        if self.is_leaf :
            self.value = value
        else :
            self.child:Dict[str,'DictTree'] = src if src is not None else {}

    def flatten(self) -> List[Tuple[List[str], Any]]:
        res = []
        def recurse(node:DictTree, path:List[str]) :
            if node.is_leaf :
                res.append((path, node.value))
            else :
                for key, value in node.child.items() :
                    recurse(value, path+[key])
        recurse(self, [])
        return res

    def find_node(self, path:List[str]) -> Union['DictTree',None]:
        if len(path) == 0 :
            return self
        elif path[0] not in self.child :
            return None
        else :
            return self.child[path[0]].find_node(path[1:])
    
    def add_node(self, path:List[str], default_value=None) -> 'DictTree':
        if len(path) == 0 :
            return self
        elif path[0] not in self.child :
            self.child[path[0]] = DictTree(is_leaf=(len(path)==1), value=default_value)
        
        return self.child[path[0]].add_node(path[1:], default_value)

    def _to_str(self, name:str, prefix:List[str], is_last:bool) -> List[str]:
        res = []
        prefix = prefix[:]
        res.extend(prefix + ['└── ' if is_last else '├── ', name])
        prefix.append('    ' if is_last else '│   ')

        if self.is_leaf :
            res.extend([': ', str(self.value), '\n'])
        else :
            res.extend(['\n'])
            for idx, (key, val) in enumerate(self.child.items()) :
                res.extend(val._to_str(key, prefix, idx==len(self.child)-1))
        return res
    
    def to_str(self, name:str=None) -> str:
        if self.is_leaf :
            raise Exception('Can\'t call to_str in leaf node.')
        
        res = []
        if name is not None :
            res.extend([name, '\n'])
        for idx, (key, val) in enumerate(self.child.items()) :
            res.extend(val._to_str(key, [], idx==len(self.child)-1))
        return ''.join(res)


import torch as th
import numpy as np
from abc import abstractmethod

class _LogData :
    @abstractmethod
    def compute(self) :
        pass
    @abstractmethod
    def __str__(self) :
        pass
    @abstractmethod
    def merge(self, data:'_LogData') :
        pass
    @abstractmethod
    def clear(self) :
        pass

class _LogDataFloat(_LogData) :
    def __init__(self, data:List[Any], fmt:str='%f'):
        self.data = []
        for x in data :
            if isinstance(x, (float, int, np.floating, np.integer)) :
                self.data.append(float(x))
            elif isinstance(x, np.ndarray) :
                self.data.append(float(x.item()))
            elif isinstance(x, th.Tensor) :
                self.data.append(float(x.item()))
            else :
                raise Exception('Not supporting type.')
        self.fmt = fmt

    def compute(self) :
        return sum(self.data) / len(self.data) if len(self.data) > 0 else None
    
    def __str__(self) :
        x = self.compute()
        if isinstance(x, float) :
            return self.fmt%x
        else :
            return 'Empty'
    
    def merge(self, data:'_LogDataFloat') :
        self.data.extend(data.data)

    def clear(self) :
        self.data.clear()


DEFAULT_FORMAT = '%.3f'

class _LogDataStr(_LogData) :
    def __init__(self, data:str) :
        if not isinstance(data, str) :
            raise Exception('Not supporting type.')
        self.data = data

    def compute(self) :
        return self.data

    def __str__(self) :
        return self.compute()
    
    def merge(self, data:'_LogDataStr') :
        self.data = data.data

    def clear(self) :
        self.data = ''

def _to_log_data(x) :
    if isinstance(x, str) :
        return _LogDataStr(x)
    else :
        return _LogDataFloat([x], DEFAULT_FORMAT)


class LogTree :
    def __init__(self) :
        self.tree = DictTree(is_leaf=False)

    @staticmethod
    def _preprocess_path(path:str) :
        return list(filter(None, map(str.strip, path.split('/'))))

    def push(self, path:str, data:Any) :
        path = self._preprocess_path(path)
        node = self.tree.find_node(path)
        if node is not None :
            node.value.merge(_to_log_data(data))
        else :
            node = self.tree.add_node(path, _to_log_data(data))

    def push_dict(self, data:Dict[str,Any]) :
        for key, value in data.items() :
            self.push(key, value)

    def push_tree(self, data:Dict[str,dict|Any]) :
        def recurse(node:dict, path:List[str]) :
            for key, val in node.items() :
                if not isinstance(key, str) :
                    continue
                if isinstance(val, dict) :
                    recurse(val, path+[key])
                else :
                    self.push('/'.join(path+[key]), val)
        recurse(data, [])

    def flatten(self) -> Dict[str, float|str]:
        raw_data = self.tree.flatten()
        data = {}
        for path, value in raw_data :
            data['/'.join(path)] = value.compute()
        return data
    
    def __str__(self) :
        return self.tree.to_str()
    
    def clear_data(self) :
        def recurse(node:DictTree) :
            if node.is_leaf :
                node.value.clear()
            else :
                for next in node.child.values() :
                    recurse(next)
        recurse(self.tree)

    def clear_tree(self) :
        self.tree = DictTree(is_leaf=False)
