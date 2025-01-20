import torch as th
from simple_rl.env.env import BaseEnv


class IsaacEnvWrapper(BaseEnv) :
    def __init__(self, env, reward_scale:float=1.0) :
        self.env = env
        self.device = env.device
        self.n_env = env.num_envs
        try :
            self.n_obs = env.observation_space['policy'].shape[1]
        except :
            self.n_obs = env.observation_space.shape[1]
        self.n_action = env.action_space.shape[1]
        self.reward_scale = reward_scale

    def reset(self) :
        obs, info = self.env.reset()
        obs = obs['policy']
        obs = obs.view(self.n_env,self.n_obs)
        return obs, info

    def step(self, action) :
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs['policy']
        obs = obs.view(self.n_env,self.n_obs)
        reward = reward.view(self.n_env,1) * self.reward_scale
        terminated = terminated.view(self.n_env,1)
        truncated = truncated.view(self.n_env,1)
        return obs, reward, terminated, truncated, info

    def close(self) :
        return self.env.close()


class GymEnvWrapper(BaseEnv) :
    def __init__(self, env, device:th.device, reward_scale:float=1.0) :
        self.env = env
        self.device = device
        self.n_env = 1
        self.n_obs = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.shape[0]
        self.reward_scale = reward_scale

    def _float(self, x) :
        return th.tensor(x, dtype=th.float32, device=self.device)

    def _bool(self, x) :
        return th.tensor(x, dtype=th.bool, device=self.device)

    def reset(self) :
        obs, info = self.env.reset()
        obs = self._float(obs).view(self.n_env,self.n_obs)
        return obs, info

    def step(self, action:th.Tensor) :
        action = action.cpu().numpy().reshape(self.env.action_space.shape)
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self._float(obs).view(self.n_env,self.n_obs)
        reward = self._float(reward).view(self.n_env,1) * self.reward_scale
        terminated = self._bool(terminated).view(self.n_env,1)
        truncated = self._bool(truncated).view(self.n_env,1)

        if terminated.item() or truncated.item() :
            obs, info = self.reset()
        return obs, reward, terminated, truncated, info

    def close(self) :
        return self.env.close()


# TODO
class GymVecEnvWrapper(BaseEnv) :
    pass
