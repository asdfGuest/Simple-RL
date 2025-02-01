import torch as th
from typing import Dict

from simple_rl.env import EnvSpec
from simple_rl.algorithms.ppo.buffer import PPOBuffer
from simple_rl.algorithms.ppo.config import PPOCfg
from simple_rl.modules.modules import BaseActorCritic
from simple_rl.modules.normalizer import Normalizer


class PPO:
    def __init__(
            self,
            env_spec:EnvSpec,
            actor_critic:BaseActorCritic,
            cfg:PPOCfg,
        ) :
        self.env_spec = env_spec
        self.actor_critic = actor_critic
        self.cfg = cfg

        self.learning_rate = self.cfg.learning_rate
        self.buffer = PPOBuffer(self.cfg)
        self.preprocessor = Normalizer((self.env_spec.n_obs,)) if self.cfg.normalize_observation else th.nn.Identity()
        
        self.actor_critic.to(self.env_spec.device)
        self.preprocessor.to(self.env_spec.device)
        
        self.optimizer = th.optim.Adam(
            params=self.actor_critic.parameters(),
            lr=self.learning_rate
        )

    def save(self, path:str) :
        th.save({
            'learning_rate' : self.learning_rate,
            'actor_critic_state_dict' : self.actor_critic.state_dict(),
            'preprocessor_state_dict' : self.preprocessor.state_dict(),
            'optimizer_state_dict' : self.optimizer.state_dict(),
        }, path)

    def load(self, path:str) :
        state_dict = th.load(path, map_location=self.env_spec.device)

        self.learning_rate = state_dict['learning_rate']
        self.actor_critic.load_state_dict(state_dict['actor_critic_state_dict'])
        self.preprocessor.load_state_dict(state_dict['preprocessor_state_dict'])
        
        self.optimizer = th.optim.Adam(
            params=self.actor_critic.parameters(),
            lr=self.learning_rate
        )
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])

    def preprocess(self, obs_raw:th.Tensor, update:bool=False) :
        kwargs = {'update':update} if self.cfg.normalize_observation else {}
        return self.preprocessor(obs_raw, **kwargs)

    def act(self, obs_raw:th.Tensor, deterministic=False) :
        obs = self.preprocess(obs_raw)
        with th.no_grad():
            self._last_act_pdf = self.actor_critic.policy.compute(obs)
            return self._last_act_pdf.mean if deterministic else self._last_act_pdf.sample()

    def collect_sample(
            self,
            obs_raw:th.Tensor,
            next_obs_raw:th.Tensor,
            action:th.Tensor,
            reward:th.Tensor,
            terminated:th.Tensor,
            truncated:th.Tensor,
        ) :
        obs = self.preprocess(obs_raw, update=True)
        with th.no_grad():
            pdf = self._last_act_pdf
            action_logprob = pdf.log_prob(action)
            value = self.actor_critic.value.compute(obs)
            reward = reward + truncated * self.cfg.gamma * value

        self.buffer.push(
            obs=obs,
            action=action,
            action_mean=pdf.mean,
            action_std=pdf.std,
            action_logprob=action_logprob,
            reward=reward,
            done=terminated|truncated,
            value=value
        )
        self.next_obs_raw = next_obs_raw.detach()

    def _update_learning_rate(self, policy_kld:float) :
        desired_kl = self.cfg.desired_kl

        if desired_kl is not None :

            if policy_kld > desired_kl * 2.0 :
                self.learning_rate = max(self.learning_rate / 1.5, 1e-5)
            elif policy_kld < desired_kl / 2.0 :
                self.learning_rate = min(self.learning_rate * 1.5, 1e-2)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate

    def update(self, train_rate:float) -> Dict[str,float]:

        _wrapper = lambda x: x if isinstance(x, float) or x is None else x(train_rate)
        ratio_clip_param = _wrapper(self.cfg.ratio_clip_param)
        value_clip_param = _wrapper(self.cfg.value_clip_param)
        entropy_loss_coeff = _wrapper(self.cfg.entropy_loss_coeff)
        
        info = {
            'PPO/policy_loss' : [],
            'PPO/value_loss' : [],
            'PPO/entropy_bonus' : [],
        }
        if self.cfg.desired_kl is not None :
            info['PPO/learning_rate'] = []

        with th.no_grad() :
            next_obs = self.preprocess(self.next_obs_raw)
            next_value = self.actor_critic.value.compute(next_obs)
        self.buffer.compute_gae(next_value)
        
        for minibatchs in self.buffer.get_minibatch_generator() :
            # get minibatch
            obs, action_old, *pdf_old, action_old_logprob, value_old, advantage, returns = minibatchs
            pdf_old = self.actor_critic.policy.get_pdf_cls()(*pdf_old)

            if self.cfg.normalize_advantage :
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            pdf = self.actor_critic.policy.compute(obs)
            # adaptive learning rate
            if self.cfg.desired_kl is not None :
                with th.no_grad() :
                    policy_kld = self.actor_critic.policy.get_pdf_cls().kld(pdf_old, pdf).mean().item()
                self._update_learning_rate(policy_kld)
            # policy loss
            action_logprob = pdf.log_prob(action_old)
            ratio = th.exp(action_logprob - action_old_logprob)
            ratio_clip = th.clip(ratio, 1.-ratio_clip_param, 1.+ratio_clip_param)
            policy_loss = th.mean(-th.min(ratio * advantage, ratio_clip * advantage))
            # value loss
            value = self.actor_critic.value.compute(obs)
            if value_clip_param is not None :
                value_clip = value_old + th.clip(value-value_old, -value_clip_param, value_clip_param)
                value_loss = th.mean(th.max(
                    (returns - value)**2,
                    (returns - value_clip)**2
                ))
            else :
                value_loss = th.mean((value - returns)**2)
            # entropy bonus
            entropy_bonus = th.mean(pdf.entropy())
            # final loss
            loss = policy_loss + value_loss * self.cfg.value_loss_coeff - entropy_bonus * entropy_loss_coeff
            # optimization
            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.grad_norm_clip is not None :
                th.nn.utils.clip_grad_norm_(self.actor_critic.policy.parameters(), self.cfg.grad_norm_clip)
            self.optimizer.step()
            # logging
            if self.cfg.desired_kl is not None :
                info['PPO/learning_rate'].append(self.learning_rate)
            info['PPO/policy_loss'].append(policy_loss.item())
            info['PPO/value_loss'].append(value_loss.item())
            info['PPO/entropy_bonus'].append(entropy_bonus.item())
        
        self.buffer.clear()

        _mean = lambda x: sum(x) / len(x)
        for key in info :
            info[key] = _mean(info[key])
        return info
