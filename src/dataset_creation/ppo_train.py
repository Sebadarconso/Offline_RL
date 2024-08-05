import gymnasium as gym 
import ale_py
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def main():

    env_name = 'Walker2d-v2'

    vec_env = make_vec_env(env_name)
    # model = PPO('MlpPolicy', vec_env, verbose=1)
    # model.learn(total_timesteps=1000000)
    # model.save(f"/home/server/Desktop/sebastiano/offline_rl/src/dataset_creation/model_chkpt/ppo_{env_name}")

    # del model ## remove to demonstrate saving and loading 
    model = PPO.load(f"/home/server/Desktop/sebastiano/offline_rl/src/dataset_creation/model_chkpt/ppo_{env_name}")

    frames = []

    obs = vec_env.reset()

    for _ in tqdm(range(5000)):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        frame = vec_env.render(mode='rgb_array')
        frames.append(frame)

    print("Simulation ended...")
    print("Creating video...")
    pil_images = [Image.fromarray(frame) for frame in frames]
    pil_images[0].save('/home/server/Desktop/sebastiano/offline_rl/src/dataset_creation/output.gif', save_all=True, append_images=pil_images[1:], loop=0, duration=40)

    
if __name__ == '__main__':
    main()