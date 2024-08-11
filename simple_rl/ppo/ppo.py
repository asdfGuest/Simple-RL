from __future__ import annotations

import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from typing import Union, List, Tuple, Type
from collections import namedtuple

from simple_rl.ppo.policy import PolicyNet, ValueNet
from simple_rl.common.logger import Logger
from simple_rl.common.callback import BaseCallback, ListCallback
from simple_rl.common.scheduler import Scheduler, to_scheduler
from simple_rl.common.utils import floatTensor, boolTensor, toFloat

from stable_baselines3.common.vec_env import VecEnv



class Hyperparameter() :
    def __init__(
            self,
            lr:Union[float, Scheduler]=0.0003,
            n_steps:int=2048,
            batch_size:int=64,
            n_epochs:int=10,
            gamma:float=0.99,
            gae_lambda:float=0.95,
            clip:Union[float,Scheduler]=0.2,
            ent_coef:float=0.0,
            vf_coef:float=0.5,
            max_grad_norm:Union[float,None]=0.5
            ) :
        self.lr:Scheduler = to_scheduler(toFloat(lr))
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = toFloat(gamma)
        self.gae_lambda = toFloat(gae_lambda)
        self.clip:Scheduler = to_scheduler(toFloat(clip))
        self.ent_coef = toFloat(ent_coef)
        self.vf_coef = toFloat(vf_coef)
        self.max_grad_norm = toFloat(max_grad_norm)



PPOData = namedtuple('PPOData', ['s', 'a', 'log_prob', 'g', 'adv'])
'''
s : (state dim,)
a : (action dim,)
log_prob : (1,)
g : (1,)
adv : (1,)
'''



class Trajectory() :
    def __init__(self) :
        self.state = []
        self.action = []
        self.log_prob = []
        self.reward = []
        self.value = []
    
    def push(
            self,
            state:th.Tensor,
            action:th.Tensor,
            log_prob:th.Tensor,
            reward:th.Tensor,
            value:th.Tensor
            ) :
        self.state.append(state)
        self.action.append(action)
        self.log_prob.append(log_prob)
        self.reward.append(reward)
        self.value.append(value)

    @property
    def length(self) :
        return len(self.state)
    
    def clear(self) :
        self.state.clear()
        self.action.clear()
        self.log_prob.clear()
        self.reward.clear()
        self.value.clear()

    def get_ppo_data(self, gamma:float, gae_lambda:float, value_terminal:th.Tensor) -> List[Type[PPOData]]:
        
        self.value.append(value_terminal)
        adv_list = [None] * self.length
        g_list = [None] * self.length

        for k in range(self.length-1, -1, -1) :
            delta = self.reward[k] + gamma * self.value[k+1] - self.value[k]
            adv_list[k] = delta + gamma * gae_lambda * (adv_list[k+1] if k+1<len(adv_list) else 0.0)
            g_list[k] = adv_list[k] + self.value[k]
        
        data = []
        for k in range(self.length) :
            data.append(PPOData(
                s=self.state[k],
                a=self.action[k],
                log_prob=self.log_prob[k],
                g=g_list[k],
                adv=adv_list[k]
            ))
        
        return data



class PPOBuffer() :
    def __init__(self, data:Union[List[Type[PPOData]],None]=None) :
        self.buffer:List[PPOData] = [] if data is None else data

    def __len__(self) :
        return len(self.buffer)
    
    def __getitem__(self, index:Union[int,Tuple[int],List[int]]) -> PPOData:
        
        if isinstance(index, int) :
            return self.buffer[index]
        
        elif isinstance(index, tuple) or isinstance(index, list) :
            s, a, log_prob, g, adv = [], [], [], [], []

            for data in [self.buffer[idx] for idx in index] :
                s.append(data.s)
                a.append(data.a)
                log_prob.append(data.log_prob)
                g.append(data.g)
                adv.append(data.adv)

            s = th.stack(s)
            a = th.stack(a)
            log_prob = th.stack(log_prob)
            g = th.stack(g)
            adv = th.stack(adv)
            
            return PPOData(s, a, log_prob, g, adv)
    
    def get(self, batch_size:int) :
        indices = th.randperm(self.__len__()).tolist()
        batched_data = []
        
        for start in range(0, self.__len__(), batch_size) :
            batched_data.append(self[indices[start:start+batch_size]])
        
        return batched_data

    def push(self, x:Union[List[PPOData],PPOData]) :
        if isinstance(x, list) :
            self.buffer += x
        else :
            self.buffer.append(x)
    
    def clear(self) :
        self.buffer.clear()



class PPO() :

    def __init__(
            self,
            env:VecEnv,
            policy_net_cls=PolicyNet,
            policy_net_kwargs:dict=dict(),
            value_net_cls=ValueNet,
            value_net_kwargs:dict=dict(),
            optim_kwargs:dict=dict(),
            hp:Hyperparameter=Hyperparameter()
            ) :
        self.env = env
        self.hp = hp

        self.action_clip_low = th.tensor(self.env.action_space.low, dtype=th.float32)
        self.action_clip_high = th.tensor(self.env.action_space.high, dtype=th.float32)

        self.logger = Logger()
        self.logger.register_key('episode/r')
        self.logger.register_key('episode/l')
        self.logger.register_key('episode/t')
        self.logger.register_key('train/lr')
        self.logger.register_key('train/policy_loss')
        self.logger.register_key('train/value_loss')
        self.logger.register_key('train/entropy')
        self.logger.register_key('train/clip_rate')
        self.logger.register_key('train/action_std')
        # build networks
        self.policy_net = policy_net_cls(
            obs_dim = env.observation_space.shape[0],
            action_dim = env.action_space.shape[0],
            **policy_net_kwargs
        )
        self.value_net = value_net_cls(
            obs_dim = env.observation_space.shape[0],
            **value_net_kwargs
        )
        # make optimizer
        self.model_parameters = list(self.policy_net.parameters()) + list(self.value_net.parameters())
        self.optimizer = Adam(
            params=self.model_parameters,
            lr=hp.lr(0.0),
            eps=1e-5,
            **optim_kwargs
        )
    

    def _optimize(self, train_rate:float, ppo_buffer:PPOBuffer) -> Tuple[float,float,float]:
        # set learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.hp.lr(train_rate)
        # optimize
        policy_loss_list = []
        value_loss_list = []
        ent_bonus_list = []
        clip_rate_list = []
        
        clip = self.hp.clip(train_rate)
        for epoch in range(1,self.hp.n_epochs+1) :
            for batch in ppo_buffer.get(self.hp.batch_size) :
                batch = PPOData(*batch)
                # predict
                action, log_prob, entropy = self.policy_net(batch.s, batch.a)
                value = self.value_net(batch.s)
                # policy loss
                ratio = (log_prob - batch.log_prob).exp()
                policy_loss = -th.min(
                    ratio * batch.adv,
                    th.clip(ratio, 1.0-clip, 1.0+clip) * batch.adv
                ).mean()
                clip_rate = ((ratio-1.0).abs() > clip).sum() / ratio.numel()
                # value loss
                value_loss = F.mse_loss(value, batch.g)
                # entropy bonus
                ent_bonus = entropy.mean()
                # loss
                loss = policy_loss + value_loss * self.hp.vf_coef - ent_bonus * self.hp.ent_coef
                # update
                self.optimizer.zero_grad()
                loss.backward()
                if self.hp.max_grad_norm != None :
                    clip_grad_norm_(self.model_parameters, self.hp.max_grad_norm)
                self.optimizer.step()
                # logging
                policy_loss_list.append(policy_loss.item())
                value_loss_list.append(value_loss.item())
                ent_bonus_list.append(ent_bonus.item())
                clip_rate_list.append(clip_rate.item())
        
        return policy_loss_list, value_loss_list, ent_bonus_list, clip_rate_list
    
    # TODO : resume train
    def train(self, total_timesteps:int, callback:BaseCallback=None, verbose:int=1) :
        
        callback = callback if callback != None else ListCallback([])

        # _on_training_start
        callback._on_training_start(self)
        # calculate total train steps
        total_steps = (total_timesteps + self.env.num_envs - 1) // self.env.num_envs
        
        # initialize buffer
        traj_buff:list[Trajectory] = [Trajectory() for _ in range(self.env.num_envs)]
        ppo_buff:PPOBuffer = PPOBuffer()
        
        # initialize observation
        obs = floatTensor(self.env.reset())

        for step in range(1, total_steps+1) :
            train_rate = step / total_steps
            timestep = step * self.env.num_envs
            # sample action
            with th.no_grad() :
                action, log_prob, _ = self.policy_net(obs)
                value = self.value_net(obs)
            # clip action for correct range
            clipped_action = action.clip(self.action_clip_low, self.action_clip_high)
            # environment step
            obs_next, reward, done, info = self.env.step(clipped_action.numpy())
            obs_next = floatTensor(obs_next)
            reward = floatTensor(reward)
            done = boolTensor(done)

            # collect rollouts
            for k in range(self.env.num_envs) :
                
                traj_buff[k].push(obs[k], action[k], log_prob[k], reward[k], value[k])

                if done[k] :
                    obs_terminal = floatTensor(info[k]['terminal_observation'])
                    if info[k]['TimeLimit.truncated'] :
                        with th.no_grad() :
                            value_terminal = self.value_net(obs_terminal)
                    else :
                        value_terminal = th.tensor([0.0], dtype=th.float32)
                    
                    ppo_buff.push(
                        traj_buff[k].get_ppo_data(
                            gamma=self.hp.gamma,
                            gae_lambda=self.hp.gae_lambda,
                            value_terminal=value_terminal
                        )
                    )
                    traj_buff[k].clear()
                    
                    # logging
                    self.logger.add('episode/r', timestep, info[k]['episode']['r'])
                    self.logger.add('episode/l', timestep, info[k]['episode']['l'])
                    self.logger.add('episode/t', timestep, info[k]['episode']['t'])
            
            # tranition observation
            obs = obs_next
            
            if step % self.hp.n_steps == 0 :
                # truncate rollouts and collect
                for k in range(self.env.num_envs) :
                    with th.no_grad() :
                        value_terminal = self.value_net(obs[k])
                    ppo_buff.push(
                        traj_buff[k].get_ppo_data(
                            gamma=self.hp.gamma,
                            gae_lambda=self.hp.gae_lambda,
                            value_terminal=value_terminal
                        )
                    )
                    traj_buff[k].clear()
                # optimize surrogate objective
                policy_loss, value_loss, entropy, clip_rate = self._optimize(train_rate, ppo_buff)
                ppo_buff.clear()
                
                # logging
                mean = lambda x : sum(x) / len(x)

                action_std = self.policy_net.std_param.detach().exp().tolist()
                self.logger.add('train/lr', timestep, self.hp.lr(train_rate))
                self.logger.add('train/policy_loss', timestep, mean(policy_loss))
                self.logger.add('train/value_loss', timestep, mean(value_loss))
                self.logger.add('train/entropy', timestep, mean(entropy))
                self.logger.add('train/clip_rate', timestep, mean(clip_rate))
                self.logger.add('train/action_std', timestep, mean(action_std))
                
                if verbose == 1 :
                    print(
                        '| %3.1f%% | policy %+7.2f | value %6.2f | entropy %4.2f | clip %4.2f | std %s |'%(
                            train_rate * 100.0,
                            self.logger.raw('train/policy_loss')[1][-1],
                            self.logger.raw('train/value_loss')[1][-1],
                            self.logger.raw('train/entropy')[1][-1],
                            self.logger.raw('train/clip_rate')[1][-1],
                            '['+('%4.2f '*len(action_std))%(*action_std,)+'\b'+']'
                        )
                    )
            
            # _on_step
            step_data = dict(
                total_timesteps=total_timesteps,
                timestep=timestep,
                total_steps=total_steps,
                step=step,
                train_rate=train_rate
            )
            callback._on_step(self, step_data)

        # _on_training_end
        callback._on_training_end(self)


    def sample_action(self, obs:th.Tensor, deterministic:bool=True) :
        with th.no_grad() :
            action, log_prob, entropy = self.policy_net(obs, deterministic=deterministic)
        clipped_action = action.clip(self.action_clip_low, self.action_clip_high)
        return clipped_action
    

    def save(self, path:str) :
        pass
    

    @staticmethod
    def load(path:str) :
        pass