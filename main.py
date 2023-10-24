# This is a sample Python script.
import gymnasium as gym
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from env import Pendulum
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from stable_baselines3 import PPO
import os
import envs
def start():
    env = Pendulum()
    model = PPO('MlpPolicy', env)
    model.learn(total_timesteps=2_000, progress_bar=True)

    model_dir = "models/"
    os.makedirs(model_dir, exist_ok=True)

    # 定义回调函数
    checkpoint_callback = CheckpointCallback(save_freq=100, save_path=model_dir, verbose=1)
    eval_callback = EvalCallback(env, callback_on_new_best=checkpoint_callback, verbose=1)

    # 训练模型并使用回调函数
    model.learn(total_timesteps=2000, callback=eval_callback, progress_bar=True)

    # 可视化训练结果
    results_plotter.plot_results(model_dir, num_timesteps=10,x_axis="time", task_name="test")
    evaluate_policy(model, env, n_eval_episodes=20)
# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    env = gym.make('envs/obs-v0', size=10, obs_num=20, render_mode="human", seed=4)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
