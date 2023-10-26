import numpy as np
import numpy.random
import pygame
import gymnasium as gym
from gymnasium import spaces
import torch


class Obsworld(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10, obs_num=None, seed=None):
        super().__init__()
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.obs_num = obs_num
        self.seed = seed
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.steps = 0
        self.arrive_count = 0
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                # "no_obs": spaces.Discrete(4),
                # "last_actions": spaces.MultiDiscrete([4, 4, 4, 4, 4])
            }
        )  # set the observation_space type dict,tuple can be space too

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)
        self.obstacle = np.zeros(shape=(self.obs_num, 2))
        self.is_obs = np.zeros(shape=(self.size, self.size))

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self.action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.agent_location = np.random.randint(low=0, high=self.size, size=2)
        self.start_pos = self.agent_location
        self.target_location = self.generate_random_target()

        # We will sample the target's location randomly until it does not coincide with the agent's location

        self.is_obs = np.zeros(shape=(size, size))
        self.generate_obstacles()

    def _get_obs(self):
        return {"agent": self.agent_location, "target": self.target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self.agent_location - self.target_location, ord=1
            )
        }

    def to_human(self):
        self.render_mode = "human"

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset()
        self.agent_location = self.start_pos
        # Choose the agent's location uniformly at random

        observation = self._get_obs()
        info = self._get_info()
        self.steps = 0
        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def generate_random_target(self):
        while True:
            tgt = np.random.randint(0, self.size, 2)
            if not np.array_equal(tgt, self.start_pos) and tgt not in self.obstacle:
                return tgt

    def get_observation(self):
        return {"agent": self.agent_location, "target": self.target_location}

    def generate_obstacles(self):
        self.obstacle = np.random.randint(low=0, high=self.size, size=(self.obs_num, 2))
        for i in range(self.obs_num):
            while np.array_equal(
                self.obstacle[i], self.agent_location
            ) or np.array_equal(self.obstacle[i], self.target_location):
                self.obstacle[i] = np.random.randint(low=0, high=self.size, size=2)
        for obs in self.obstacle:
            self.is_obs[obs[0], obs[1]] = 1

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self.action_to_direction[action]

        obs, info, reward, new_location = self.act(action=action)

        reward -= self.steps
        terminated = np.array_equal(self.agent_location, self.target_location)

        if self.render_mode == "human":
            self._render_frame()
        self.steps += 1
        truncated = False
        if terminated:
            self.arrive_count += 1
        if self.steps >= 200:
            truncated = True
        print(self.steps, self.arrive_count)
        return obs, reward, terminated, truncated, info

    def legal(self, space):
        space = space.astype(np.int64)
        r1 = space >= 0
        if not np.all(r1):
            return False
        r2 = space < self.size
        if not np.all(r2):
            return False
        r3 = self.is_obs[space[0], space[1]] == 0
        return r3

    def act(self, action):
        # print("dir",self.action_to_direction[action],"state:",state,"new_state:",new_state)
        new_location = self.agent_location + self.action_to_direction[action]
        # print(self.agent_location, self.legal(new_location), new_location)
        reward = 0

        if not self.legal(new_location):
            reward -= 30
            new_location = self.agent_location

        if np.array_equal(new_location, self.target_location):
            reward += 1000
        obs = self.get_observation()
        info = self._get_info()
        # reward -= abs(3 * info["distance"])
        self.agent_location = new_location
        return obs, info, reward, new_location

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # 白色背景
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.target_location[0],
                pix_square_size * self.target_location[1],
                pix_square_size,
                pix_square_size,
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 255, 0),
            pygame.Rect(
                pix_square_size * self.start_pos[0],
                pix_square_size * self.start_pos[1],
                pix_square_size,
                pix_square_size,
            ),
        )
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self.target_location[0],
                pix_square_size * self.target_location[1],
                pix_square_size,
                pix_square_size,
            ),
        )
        for obs in self.obstacle:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * obs[0],
                    pix_square_size * obs[1],
                    pix_square_size,
                    pix_square_size,
                ),
            )
        # Now we draw the agent
        pix_pos = (
            (self.agent_location[0] + 0.5) * pix_square_size,
            (self.agent_location[1] + 0.5) * pix_square_size,
        )
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            pix_pos,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
