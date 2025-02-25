import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py

gym.register_envs(ale_py)

# definindo ambiente principal
env = make_atari_env("SpaceInvadersNoFrameskip-v4", n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)

# definindo ambiente de avaliação
eval_env = make_atari_env("SpaceInvadersNoFrameskip-v4", n_envs=4, seed=0)
eval_env = VecFrameStack(eval_env, n_stack=4)

# Agente DQN
model = DQN(
    'CnnPolicy',                # Política baseada em CNN
    env,
    learning_rate=3e-4,         # Taxa de aprendizado
    buffer_size=50000,          # Tamanho do buffer de replay
    learning_starts=0,          # Número de passos antes de começar a aprender
    batch_size=64,              # Tamanho do lote para atualização
    tau=1.0,                    # Parâmetro de atualização do target network
    gamma=0.99,                 # Fator de desconto
    train_freq=4,               # Frequência de atualização da rede
    gradient_steps=1,           # Número de passos de gradiente por atualização
    target_update_interval=5000,# Intervalo de atualização do target network
    exploration_fraction=0.2,   # Fração do tempo para explorar
    exploration_final_eps=0.01, # Valor final de epsilon para exploração
    tensorboard_log="./tensorboard_logs/",
    verbose=1,
    device="cuda"
)