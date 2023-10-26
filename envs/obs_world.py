import numpy as np

from envs.obstacle_gridworld import Obsworld


class obs_world(Obsworld):
    def __init__(self, render_mode=None, size=10, obs_num=None, seed=None):
        super().__init__(render_mode=render_mode, size=size, obs_num=obs_num, seed=seed)

    @property
    def temperature(self):
        return 10 * np.exp(-self.steps / 50)

    @property
    def _get_obs(self):
        return super()._get_obs()

    @property
    def _get_info(self):
        return super()._get_info()

    def act(self, action):

        new_loc = self.agent_location + self.action_to_direction[action]
        terminated = False
        if not self.legal(new_loc):
            reward = -10
            new_loc = self.agent_location
        elif np.array_equal(new_loc, self.target_location):
            reward = 100
            self.arrive_count += 1
            terminated = True
        else:
            # 计算曼哈顿距离
            dist = sum(abs(new_loc - self.target_location))
            # 距离奖励
            reward_dist = 1 / dist * self.temperature
            reward = reward_dist - 1

        self.agent_location = new_loc
        obs = self._get_obs
        info = self._get_info
        return obs, info, reward, terminated

    def step(self, action):
        # if np.random.random() < 0.05:
        #     # 5%概率重置agent位置
        #     self.agent_location = self.reset_agent()
        # print(self.steps, self.arrive_count)
        obs, info, reward, terminated = self.act(action)
        truncated = False
        self.steps += 1
        if self.steps >= 500:
            truncated = True
        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info
