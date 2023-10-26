import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
# from callbacks.easy_callback_log import BestModelCallback
from callbacks.early_stop_withlog import BestModelCallback
register(
    id="envs/obs-v1",
    entry_point="obs_world:obs_world"
)

def test(model, env):
    state, _ = env.reset()
    for i in range(300):
        state = env.unwrapped.get_observation()
        act = int(model.predict(state)[0])
        obs, reward, terminated, _, info = env.step(act)
        if terminated:
            env.reset()

env = gym.make('envs/obs-v1', render_mode="human", size=16, obs_num=100, seed=4)
env = Monitor(env, 'Monitor_data')
# model = PPO.load('model_save/best1')
# test(model,env)