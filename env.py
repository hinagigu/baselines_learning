import gymnasium as gym


class Pendulum(gym.Wrapper):
    def __init__(self):
        env = gym.make('Pendulum-v1')
        super().__init__(env)
        self.env = env

    def reset(
        self, *, seed: int | None = None, options: dict[str, int] | None = None
    ) -> tuple[int, dict[str, int]]:
        state = self.env.reset()
        return state
    def step(self, action):
        state, reward, done, _, info = self.env.step(action)
        return state, reward, done, _,  info

