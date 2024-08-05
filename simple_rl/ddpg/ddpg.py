from __future__ import annotations

import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from typing import Union, List, Tuple
from collections import namedtuple

from simple_rl.ddpg.policy import PiNet, QNet
from simple_rl.common.noise import GaussianNoise
from simple_rl.common.logger import Logger
from simple_rl.common.callback import BaseCallback, ListCallback
from simple_rl.common.scheduler import Scheduler, to_scheduler
from simple_rl.common.utils import floatTensor, boolTensor, toFloat

from stable_baselines3.common.vec_env import VecEnv



class Hyperparameter() :
    def __init__(
            self,
            actor_lr:Union[float, Scheduler]=0.001,
            critic_lr:Union[float, Scheduler]=0.001,
            buffer_size:int=1000000,
            warmup_steps:int=100,
            batch_size:int=256,
            tau:float=0.005,
            gamma:float=0.99,
            train_freq:int=1,
            gradient_steps:int=1
            ) :
        self.actor_lr:Scheduler = to_scheduler(toFloat(actor_lr))
        self.critic_lr:Scheduler = to_scheduler(toFloat(critic_lr))
        self.buffer_size = buffer_size
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.tau = toFloat(tau)
        self.gamma = toFloat(gamma)
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps



DDPGData = namedtuple('DDPGData', ['s', 'a', 'r', 's_next', 'd'])
'''
s : (state dim,)
a : (action dim,)
r : (1,)
s_next : (state dim,)
d : (1,)
'''



class DDPGBuffer() :
    def __init__(self, capacity:int) :
        self.buffer:List[DDPGData] = []
        self.ptr = 0
        self.cap = capacity

    def __len__(self) :
        return len(self.buffer)
    
    def __getitem__(self, index:int) -> DDPGData:
        return self.buffer[index]
    
    def sample(self, batch_size:int) -> DDPGData:
        indices = th.randint(0, self.__len__(), (batch_size,))
        s, a, r, s_next, d = [], [], [], [], []

        for index in indices :
            data = self.buffer[index.item()]
            s.append(data.s)
            a.append(data.a)
            r.append(data.r)
            s_next.append(data.s_next)
            d.append(data.d)
        
        s = th.stack(s)
        a = th.stack(a)
        r = th.stack(r)
        s_next = th.stack(s_next)
        d = th.stack(d)

        return DDPGData(s, a, r, s_next, d)
    
    def push(self, x:DDPGData) :
        if self.__len__() < self.cap :
            self.buffer.append(None)
        self.buffer[self.ptr] = x
        self.ptr = (self.ptr + 1) % self.cap
    
    def clear(self) :
        self.buffer.clear()
        self.ptr = 0



class DDPG() :
    def __init__(
            self,
            env:VecEnv,
            pi_net_cls=PiNet,
            pi_net_kwargs:dict=dict(),
            q_net_cls=QNet,
            q_net_kwargs:dict=dict(),
            pi_optim_kwargs:dict=dict(),
            q_optim_kwargs:dict=dict(),
            noise_cls=GaussianNoise,
            noise_kwargs:dict=dict(),
            hp:Hyperparameter=Hyperparameter()
            ) :
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.hp = hp

        self.action_low = th.tensor(self.env.action_space.low, dtype=th.float32)
        self.action_high = th.tensor(self.env.action_space.high, dtype=th.float32)

        self.noise = noise_cls((self.env.num_envs, self.act_dim), **noise_kwargs)

        self.buffer:DDPGBuffer = DDPGBuffer(self.hp.buffer_size)

        self.logger = Logger()
        self.logger.register_key('episode/r')
        self.logger.register_key('episode/l')
        self.logger.register_key('episode/t')
        self.logger.register_key('train/actor_lr')
        self.logger.register_key('train/critic_lr')
        self.logger.register_key('train/actor_loss')
        self.logger.register_key('train/critic_loss')
        # Policy network
        self.pi_net       :PiNet = pi_net_cls(self.obs_dim, self.act_dim, **pi_net_kwargs)
        self.pi_target_net:PiNet = pi_net_cls(self.obs_dim, self.act_dim, **pi_net_kwargs)
        self.pi_target_net.load_state_dict(self.pi_net.state_dict())
        # Q network
        self.q_net       :QNet = q_net_cls(self.obs_dim, self.act_dim, **q_net_kwargs)
        self.q_target_net:QNet = q_net_cls(self.obs_dim, self.act_dim, **q_net_kwargs)
        self.q_target_net.load_state_dict(self.q_net.state_dict())
        # make optimizer
        self.pi_optimizer = Adam(self.pi_net.parameters(), lr=hp.actor_lr(0.0), **pi_optim_kwargs)
        self.q_optimizer = Adam(self.q_net.parameters(), lr=hp.critic_lr(0.0), **q_optim_kwargs)

    def _update(self, train_rate:float) -> Tuple[List[float],List[float]]:
        # set learning rate
        for param_group in self.pi_optimizer.param_groups:
            param_group['lr'] = self.hp.actor_lr(train_rate)
        for param_group in self.q_optimizer.param_groups:
            param_group['lr'] = self.hp.critic_lr(train_rate)
        
        actor_loss_list = []
        critic_loss_list = []
        
        for step in range(1,self.hp.gradient_steps+1) :
            batch = self.buffer.sample(self.hp.batch_size)
            # critic loss
            with th.no_grad() :
                q_target = self.q_target_net(batch.s_next, self.pi_target_net(batch.s_next))
            q_target = batch.r + (1.0 - batch.d) * self.hp.gamma * q_target
            q_pred = self.q_net(batch.s, batch.a)
            critic_loss = F.mse_loss(q_pred, q_target)
            # optimize critic
            self.q_optimizer.zero_grad()
            critic_loss.backward()
            self.q_optimizer.step()
            # actor loss
            actor_loss = -self.q_net(batch.s, self.pi_net(batch.s)).mean()
            # optimize actor
            self.pi_optimizer.zero_grad()
            actor_loss.backward()
            self.pi_optimizer.step()
            # update target parameter
            for param, target_param in zip(self.q_net.parameters(), self.q_target_net.parameters()) :
                target_param.data.copy_((1.0 - self.hp.tau) * target_param.data + self.hp.tau * param.data)
            for param, target_param in zip(self.pi_net.parameters(), self.pi_target_net.parameters()) :
                target_param.data.copy_((1.0 - self.hp.tau) * target_param.data + self.hp.tau * param.data)
            # logging
            actor_loss_list.append(actor_loss.item())
            critic_loss_list.append(critic_loss.item())
        
        return actor_loss_list, critic_loss_list
    
    def _warmup(self) :
        
        # initialize observation
        obs = floatTensor(self.env.reset())

        for step in range(1, self.hp.warmup_steps+1) :
            # action
            action = th.rand((self.env.num_envs, self.act_dim)) * 2.0 - 1.0
            # environment step
            obs_next, reward, done, info = self.env.step(self.scale_action(action).numpy())
            obs_next = floatTensor(obs_next)
            reward = floatTensor(reward).unsqueeze(0)
            done = boolTensor(done)
            
            # collect transition
            for k in range(self.env.num_envs) :
                
                if done[k] :
                    obs_terminal = floatTensor(info[k]['terminal_observation'])
                
                self.buffer.push(DDPGData(
                    s=obs[k],
                    a=action[k],
                    r=reward[k],
                    s_next= obs_next[k] if not done[k] else obs_terminal,
                    d=floatTensor(0.0 if not done[k] or info[k]['TimeLimit.truncated'] else 1.0).unsqueeze(0)
                ))
            
            # tranition observation
            obs = obs_next
    
    # TODO : resume train
    def train(self, total_timesteps:int, callback:BaseCallback=None) :
        
        callback = callback if callback != None else ListCallback([])

        if len(self.buffer) == 0 :
            self._warmup()
        # _on_training_start
        callback._on_training_start(self)
        # calculate total train steps
        total_steps = (total_timesteps + self.env.num_envs - 1) // self.env.num_envs
        
        # initialize observation
        obs = floatTensor(self.env.reset())

        for step in range(1, total_steps+1) :
            train_rate = step / total_steps
            timestep = step * self.env.num_envs
            # action
            with th.no_grad() :
                action = self.pi_net(obs)
            action = (action + self.noise()).clip(-1.0,1.0)
            # environment step
            obs_next, reward, done, info = self.env.step(self.scale_action(action).numpy())
            obs_next = floatTensor(obs_next)
            reward = floatTensor(reward).unsqueeze(0)
            done = boolTensor(done)
            
            # collect transition
            for k in range(self.env.num_envs) :
                
                if done[k] :
                    obs_terminal = floatTensor(info[k]['terminal_observation'])
                    # logging
                    self.logger.add('episode/r', timestep, info[k]['episode']['r'])
                    self.logger.add('episode/l', timestep, info[k]['episode']['l'])
                    self.logger.add('episode/t', timestep, info[k]['episode']['t'])
                
                self.buffer.push(DDPGData(
                    s=obs[k],
                    a=action[k],
                    r=reward[k],
                    s_next= obs_next[k] if not done[k] else obs_terminal,
                    d=floatTensor(0.0 if not done[k] or info[k]['TimeLimit.truncated'] else 1.0).unsqueeze(0)
                ))
            
            # tranition observation
            obs = obs_next

            if step % self.hp.train_freq == 0 :
                # optimize surrogate objective
                actor_loss, critic_loss = self._update(train_rate)
                # logging
                mean = lambda x : sum(x) / len(x)
                self.logger.add('train/actor_lr', timestep, self.hp.actor_lr(train_rate))
                self.logger.add('train/critic_lr', timestep, self.hp.critic_lr(train_rate))
                self.logger.add('train/actor_loss', timestep, mean(actor_loss))
                self.logger.add('train/critic_loss', timestep, mean(critic_loss))
            
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

    def scale_action(self, action:th.Tensor) :
        return (action * .5 + .5) * (self.action_high - self.action_low) + self.action_low

    def sample_action(self, obs:th.Tensor, deterministic:bool=False) :
        with th.no_grad() :
            action = self.pi_net(obs)
        return self.scale_action(action)
    
    def save(self, path:str) :
        pass
    
    @staticmethod
    def load(path:str) :
        pass