import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py

gym.register_envs(ale_py)

# configurando ambiente
env = make_atari_env("SpaceInvadersNoFrameskip-v4", n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

# definindo o caminho do modelo para avaliar
model_path = "dqn_spaceinvaders.zip"

# carregando modelo
model = DQN.load(model_path, env=env)

# avaliação
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print(f"Recompensa média: {mean_reward:.2f} +/- {std_reward:.2f}")

env.close()