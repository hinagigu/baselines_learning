import logging

# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
#
# file_handler = logging.FileHandler("training.log")
# file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
# logger.addHandler(file_handler)
#

class BestModelCallback:
    def __init__(self, model, env, check_freq=100, n_eval_episodes=10, patience=20):
        self.model = model
        self.env = env
        self.check_freq = check_freq
        self.patience = patience
        self.best_reward = -float("inf")
        self.n_eval_episodes = n_eval_episodes
        self.call_count = 0
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.file_handler = logging.FileHandler("training.log")
        self.file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(self.file_handler)
    def __call__(self, loc, global_step, *_args, **kwargs):
        self.call_count += 1
        if self.call_count % self.check_freq == 0:
            reward = self.evaluate()

            if reward > self.best_reward:
                self.best_reward = reward
                self.model.save("best_model")
                self.logger.info(
                    f"Step {self.call_count}: Best reward updated to {self.best_reward:.2f}"
                )
            else:
                self.logger.info(
                    f"Step {self.call_count}: Reward {reward:.2f} did not beat best {self.best_reward:.2f}"
                )
                self.patience -= 1
                if self.patience <= 0:
                    self.logger.info(f"Early stopped at step {global_step}!")
                    self.model.save("fixed_model")
                    return False

        return True

    def evaluate(self):
        # 评估逻辑
        reward = 0
        for _ in range(self.n_eval_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = int(self.model.predict(state)[0])
                # state, reward, done, _, terminated = self.env.step(action)
                obs, reward, terminated, truncated, info = self.env.step(action)
                if terminated:
                    self.env.reset()
                episode_reward += reward
            reward += episode_reward
        return reward


