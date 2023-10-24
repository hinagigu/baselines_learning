import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
# from callbacks.best_callback import BestModelCallback
from callbacks.early_stop_withlog import BestModelCallback
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


env = gym.make('envs/obs-v0', render_mode=None, size=16, obs_num=100, seed=4)
env = Monitor(env, 'Monitor_data')
model = PPO('MultiInputPolicy', env, verbose=0)
log_callback = BestModelCallback(model=model,env=env,check_freq=100,n_eval_episodes=10,patience=20)
model.learn(200, callback=log_callback, progress_bar=True)
# model.learn(2000, progress_bar=True)
# env.unwrapped.to_human()

# test(model,env)