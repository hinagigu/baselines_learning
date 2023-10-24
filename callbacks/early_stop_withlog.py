import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('training.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(file_handler)

class BestModelCallback:
    def __init__(self, model, env, check_freq=100,n_eval_episodes=10,patience=20):
        self.model = model
        self.env = env
        self.check_freq = check_freq
        self.patience = patience
        self.best_reward = -float('inf')
        self.n_eval_episodes = n_eval_episodes


    def __call__(self, loc, global_step, *_args, **kwargs):
        if loc % self.check_freq == 0:
            reward = self.evaluate()

            if reward > self.best_reward:
                self.best_reward = reward
                self.model.save('best_model')
                logger.info(f"Step {global_step}: Best reward updated to {self.best_reward:.2f}")
            else:
                logger.info(f"Step {global_step}: Reward {reward:.2f} did not beat best {self.best_reward:.2f}")
                self.patience -= 1
                if self.patience <= 0:
                    logger.info(f"Early stopped at step {global_step}!")
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
                state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
            reward += episode_reward
        return reward

# 训练结束后关闭handler
file_handler.close()