from __future__ import annotations

import numpy as np

from typing import List, Type, Union, Tuple, TYPE_CHECKING, Callable
from abc import ABC, abstractmethod

from simple_rl.common.evaluation import evaluate_policy

from stable_baselines3.common.vec_env import VecEnv

if TYPE_CHECKING:
    from simple_rl import PPO



class BaseCallback(ABC) :
    @abstractmethod
    def _on_training_start(self, model:PPO) :
        pass
    @abstractmethod
    def _on_step(self, model:PPO, train_rate:float) :
        pass
    @abstractmethod
    def _on_training_end(self, model:PPO) :
        pass



class ListCallback(BaseCallback) :
    def __init__(self, callbacks:List[Type[BaseCallback]]) :
        super().__init__()
        self.callbacks = callbacks

    def _on_training_start(self, model:PPO) :
        for callback in self.callbacks :
            callback._on_training_start(model)
    
    def _on_step(self, model:PPO, step_data:dict):
        for callback in self.callbacks :
            callback._on_step(model, step_data)
    
    def _on_training_end(self, model:PPO) :
        for callback in self.callbacks :
            callback._on_training_end(model)



class EvalCallback(BaseCallback) :
    def __init__(
        self,
        env:VecEnv,
        eval_steps:int,
        path:Union[str,None]=None,
        n_eval_episodes:int=4,
        deterministic: bool = True,
        verbose:int=0
    ) :
        super().__init__()
        self.env = env
        self.eval_steps = eval_steps
        self.path = path
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.verbose = verbose

        self.timestep_list = []
        self.mean_list = []
        self.std_list = []

    def _on_training_start(self, model:PPO) :
        pass
    
    def _on_step(self, model:PPO, step_data:dict):
        if step_data['step'] % self.eval_steps != 0 :
            return
        
        mean, std = evaluate_policy(
            model=model,
            env=self.env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=self.deterministic
        )
        self.timestep_list.append(step_data['timestep'])
        self.mean_list.append(mean)
        self.std_list.append(std)

        if self.path != None :
            self._save()

        if self.verbose == 1 :
            print(
                '| timestep %6d (%3.1f%%) | mean %+8.2f | std %6.2f |'%(
                    step_data['timestep'],
                    step_data['train_rate'] * 100.0,
                    mean,
                    std
                )
            )
    
    def _on_training_end(self, model:PPO) :
        pass
    
    def _save(self) :
        np.savez(
            self.path,
            timestep=self.timestep_list,
            mean=self.mean_list,
            std=self.std_list
        )

    @staticmethod
    def load(path:str) -> Tuple[int,float,float]:
        data = np.load(path)
        return data['timestep'].tolist(), data['mean'].tolist(), data['std'].tolist()



class MacroCallback(BaseCallback) :
    def __init__(self, n_steps:int, f:Callable, f_kwargs:dict=dict(), call_on_start:bool=True, call_on_end:bool=True) :
        super().__init__()
        self.n_steps = n_steps
        self.f = f
        self.f_kwargs = f_kwargs
        self.call_on_start = call_on_start
        self.call_on_end = call_on_end

    def _on_training_start(self, model) :
        if self.call_on_start :
            self.f(**self.f_kwargs)
    
    def _on_step(self, model, step_data:dict):
        if step_data['step'] % self.n_steps == 0 :
            self.f(**self.f_kwargs)
    
    def _on_training_end(self, model) :
        if self.call_on_end :
            self.f(**self.f_kwargs)