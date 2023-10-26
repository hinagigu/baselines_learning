import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
# from callbacks.easy_callback_log import BestModelCallback
from callbacks.early_stop_withlog import BestModelCallback
from gymnasium.envs.registration import register

register(
    id="envs/obs-v1",
    entry_point="envs.obs_world:obs_world"
)

def test(model, env):
    state, _ = env.reset()
    for i in range(300):
        state = env.unwrapped.get_observation()
        act = int(model.predict(state)[0])
        obs, reward, terminated, truncated, info = env.step(act)
        if terminated or truncated:
            env.reset()


env = gym.make('envs/obs-v1', render_mode="human", size=16, obs_num=100, seed=4)
env = Monitor(env, '../datafile/Monitor_data')
# model = PPO('MultiInputPolicy', env, verbose=0)
# log_callback = BestModelCallback(model=model,env=env,check_freq=100,n_eval_episodes=30,patience=1000)
# model.learn(1000, callback=log_callback, progress_bar=True)
model = PPO.load('../model_save/last_model.zip')
test(model,env)