from simple_rl.algorithms.ppo import PPO, PPOCfg
from simple_rl.modules.modules import MlpPolicy, MlpValue
from simple_rl.env.wrapper import GymEnvWrapper

import torch as th
import gymnasium as gym

env = gym.make('Hopper-v5', render_mode='human')
env = GymEnvWrapper(
    env=env,
    device=th.device('cpu'),
    reward_scale=1/100
)

policy = MlpPolicy(env.n_obs, env.n_action, 0.8, [64,64], th.nn.SiLU)
value = MlpValue(env.n_obs, [64,64], th.nn.SiLU)

cfg = PPOCfg(
    n_rollout=8192,
    n_epoch=32,
    n_minibatch=16,
    gamma=0.99,
    gae_lambda=0.95,
    learning_rate=0.0001,
    desired_kl=None,
    normalize_observation=True,
    ratio_clip_param=0.2,
    value_clip_param=None,
    grad_norm_clip=None,
    normalize_advantage=True,
    entropy_loss_coeff=0.0,
    value_loss_coeff=1.0,
)
ppo = PPO(env.spec, policy, value, cfg)
ppo.load('examples/models/hopper_model.pt')

obs, info = env.reset()
episode_length = 0
episode_reward = 0.

for _ in range(2500) :
    # sample action from deterministic policy
    with th.no_grad() :
        obs = ppo.preprocess(obs)
        action = policy.compute(obs).mean
    
    obs, reward, terminated, truncated, info = env.step(action)
    episode_length += 1
    episode_reward += reward.item()

    if terminated.item() or truncated.item() :
        print('episode done with length %d and reward %f'%(episode_length, episode_reward))
        episode_length = 0
        episode_reward = 0.

env.close()
