import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
# from callbacks.best_callback import BestModelCallback
# from callbacks.early_stop_withlog import BestModelCallback
# from callbacks.easy_callback import EasyCallBack
from callbacks.easy_callback_log import BestModelCallback
from stable_baselines3.common.evaluation import evaluate_policy
register(

    id="envs/obs-v0",
    entry_point="obstacle_gridworld:Obsworld"
)

def test(model, env):
    state, _ = env.reset()
    for i in range(300):
        state = env.unwrapped.get_observation()
        act = int(model.predict(state)[0])
        obs, reward, terminated, _, info = env.step(act)
        if terminated:
            env.reset()


env = gym.make('envs/obs-v0', render_mode="human", size=16, obs_num=100, seed=4)
# env = Monitor(env, 'Monitor_data_best1')
# model = PPO('MultiInputPolicy', env, verbose=0)
model = PPO.load('best_model.zip')
# log_callback = BestModelCallback(model=model,env=env,check_freq=100,n_eval_episodes=30,patience=100)
# model.learn(10000, callback=log_callback, progress_bar=True)
test(model,env)