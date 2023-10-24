
class BestModelCallback:
    def __init__(self, model, env, n_eval_episodes=10):
        self.model = model
        self.env = env
        self.n_eval_episodes = n_eval_episodes
        self.best_reward = -float('inf')

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
        if reward > self.best_reward:
            self.best_reward = reward
            # Save the model weights
            self.model.save("best_model_PPO")