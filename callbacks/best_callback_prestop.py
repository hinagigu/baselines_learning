class BestModelCallback:
    def __init__(self, model, env, n_eval_episodes=10, patience=100):
        self.model = model
        self.env = env
        self.n_eval_episodes = n_eval_episodes
        self.patience = patience  # 多少步reward无提升就提前终止
        self.patience_count = 0
        self.last_best_reward = -float("inf")
    def __call__(self, loc, global_step):
        # Evaluate the model
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
            # Check if the model has improved
        if reward > self.last_best_reward:
            self.last_best_reward = reward
            self.model.save("best_model_PPO")
        else:
            if abs(reward - self.last_best_reward) < 1e-3:
                self.patience_count += 1
            else:
                self.patience_count = 0
        if self.patience_count >= 10:
            print("Early stopping")
            self.model.save("best_model_PPO")
            return False  # 提前终止训练

            # Save the model weights
