from stable_baselines3.common.callbacks import BaseCallback

class QValueCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(QValueCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # obtendo o q-valor médio
        obs = self.training_env.reset()
        q_values = self.model.q_net(self.model.policy.obs_to_tensor(obs)[0])
        mean_q_value = q_values.mean().item()

        # registrando no tensorboard
        self.logger.record("mean_q_value", mean_q_value)
        
        return True