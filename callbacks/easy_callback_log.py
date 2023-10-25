import logging
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler("training.log")
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(file_handler)
#

class BestModelCallback(BaseCallback):
    def __init__(self, model, env, check_freq=100, n_eval_episodes=200, patience=20,best_path="best_model",fix_path="fixed_model"):
        super().__init__(verbose=0)
        self.model = model
        self.env = env
        self.check_freq = check_freq
        self.patience = patience
        self.best_reward = -float("inf")
        self.n_eval_episodes = n_eval_episodes
        self.call_count = 0
        self.best_model_path = best_path
        self.fix_path = fix_path
    def _on_step(self):
        self.call_count += 1
        # print("call_count",self.call_count)
        if self.call_count % self.check_freq == 0:
            reward, std = self.evaluate()
            if reward > self.best_reward:
                self.best_reward = reward
                self.model.save(self.best_model_path)
                logger.info(
                    f"Step {self.call_count}: Best reward updated to {self.best_reward:.2f}"
                )
            else:
                logger.info(
                    f"Step {self.call_count}: Reward {reward:.2f} did not beat best {self.best_reward:.2f}"
                )
                self.patience -= 1
                if self.patience <= 0:
                    logger.info(f"Early stopped at step {self.env.steps}!")
                    self.model.save(self.fix_path)
                    return False
        if self.call_count >= 10000:
            self.finish_training()
            return False
        return True

    def evaluate(self):

        # 评估逻辑
        mean_reward, std = evaluate_policy(model=self.model,env=self.env,n_eval_episodes=self.n_eval_episodes)
        return mean_reward, std
    def finish_training(self):
        file_handler.close()

