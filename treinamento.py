from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from callback import QValueCallback
from mcts import MCTS
from agente import model, env, eval_env
import numpy as np

# EvalCallback - para obter dados de recompensa média sem ruído do treinamento
eval_callback = EvalCallback(
    eval_env,
    n_eval_episodes=5,
    eval_freq=5000,
    deterministic=True,
    best_model_save_path="./logs/best_model/",
    verbose=1
)

# MetricsCallback - para obter os dados do q-valor médio
q_value_callback = QValueCallback()

# CheckpointCallback - para realizar salvamentos durante o treinamento
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./logs/",
    name_prefix="dqn_spaceinvaders",
)

# converte os dados recebidos pelo MCTS e converte para o formato do buffer de replay
def colocar_valores_no_buffer(state, action, reward, next_state, done, model):
    # converte os dados
    if not isinstance(state, np.ndarray):
        state = np.array(state, dtype=np.float32)
    if not isinstance(next_state, np.ndarray):
        next_state = np.array(next_state, dtype=np.float32)
    if not isinstance(action, np.ndarray):
        action = np.array([[action]] * model.n_envs, dtype=np.int64)
    if not isinstance(reward, (float, np.float32)):
        reward = np.array([float(reward)] * model.n_envs, dtype=np.float32)
    if not isinstance(done, bool):
        done = np.array([bool(done)] * model.n_envs)

    # reorganiza as dimensões
    state = np.transpose(state, (0, 3, 1, 2))
    next_state = np.transpose(next_state, (0, 3, 1, 2))

    # adiciona os valores ao buffer
    model.replay_buffer.add(
        obs=state,
        action=action,
        reward=reward,
        next_obs=next_state,
        done=done,
        infos=[{}] * model.n_envs
    )

# inicializando o MCTS
num_simulations = 5000
num_trajectories = 10

mcts = MCTS(env, num_simulations=num_simulations)
initial_state = env.reset()
trajectories = mcts.generate_trajectories(initial_state, num_trajectories=num_trajectories)

# adicionando as trajetórias ao buffer de replay
for state, action, reward, next_state, done in trajectories:
    colocar_valores_no_buffer(state, action, reward, next_state, done, model)

# pré-Treinamento
model.learn(total_timesteps=10_000)

# treinamento
model.learn(total_timesteps=100_000, log_interval=5000, callback=[q_value_callback, checkpoint_callback, eval_callback], progress_bar=True)

# salvando o modelo treinado
model.save("dqn_spaceinvaders_vec")

env.close()
eval_env.close()