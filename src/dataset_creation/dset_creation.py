import minari 
import gymnasium as gym 

from minari import DataCollector
from stable_baselines3 import PPO


def main():

    ## gym environment 
    env_name = 'Walker2d-v2'
    env = gym.make(env_name)
    env = DataCollector(env)

    ## model
    model = PPO.load(f'/home/server/Desktop/sebastiano/offline_rl/src/dataset_creation/model_chkpt/ppo_{env_name}.zip')

    for _ in range(100):
        state = env.reset()
        done = False

        while not done:
            action, _states = model.predict(state[0])
            obs, rew, terminated, truncated, info = env.step(action)
            done = terminated or truncated


    print("Creating dataset...")
    dataset = env.create_dataset(f'/home/server/Desktop/sebastiano/offline_rl/assets/dataset/{env_name}')
    print("Dataset created!")

if __name__ == '__main__':
    main()