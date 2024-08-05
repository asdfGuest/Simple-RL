from __future__ import annotations

import math
from typing import Tuple, TYPE_CHECKING

from simple_rl.common.utils import floatTensor, boolTensor

from stable_baselines3.common.vec_env import VecEnv

if TYPE_CHECKING:
    from simple_rl import PPO



def evaluate_policy(
    model:PPO,
    env:VecEnv,
    n_eval_episodes:int = 8,
    deterministic:bool = True
) -> Tuple[float, float]:
    
    episode_reward = []
    
    obs = floatTensor(env.reset())
    env_reward = [0.0 for _ in range(env.num_envs)]

    while True :
        
        action = model.sample_action(obs, deterministic)
        
        obs, reward, done, info = env.step(action.numpy())
        obs = floatTensor(obs)
        reward = floatTensor(reward)
        done = boolTensor(done)

        flag = False
        for k in range(env.num_envs) :
            env_reward[k] += reward[k].item()
            
            if done[k] :
                episode_reward.append(env_reward[k])
                env_reward[k] = 0.0

                if len(episode_reward) == n_eval_episodes :
                    flag = True
                    break
        if flag :
            break
    
    # calculate statistics
    mean = lambda x : sum(x) / len(x)
    
    mean_reward = mean(episode_reward)
    std_reward = math.sqrt(mean([(reward-mean_reward)**2 for reward in episode_reward]))

    return mean_reward, std_reward