import os
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = "1" ## suppresses warning about carla and flow failed imports

import argparse
import torch
import gym
import d4rl

import matplotlib.pyplot as plt
from src.testing.cql_custom import *
from moviepy.editor import ImageSequenceClip
from tqdm import tqdm

def main(opt):

    env_name = opt.env_name
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print("*"*50)
    print(f"Environment: {env_name}")
    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Max action: {max_action}")
    print("*"*50)

    if opt.algo == 'cql':
        actor = TanhGaussianPolicy(
            state_dim,
            action_dim,
            max_action,
            log_std_multiplier=1.0,
            orthogonal_init=True,
        ).to("cuda")
    else:
        ...

    actor.load_state_dict(torch.load(opt.ckpt_path)['actor'])
    actor.eval()
    print("State dict succesfully loaded!")

    
    num_episodes = opt.n_episodes
    video_frames = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = actor.act(state, device='cuda')
            next_state, reward, done, info = env.step(action)
            video_frames.append(env.render(mode='rgb_array'))
    
            state = next_state
        print(f"Reward: {reward}")

    if opt.simulation:
        ## saving video
        simulation_path = 'assets/plots/simulations/'
        filename = 'cql_' + opt.env_name +'_simulation.mp4'
        frames_per_second = 10

        clip = ImageSequenceClip(video_frames, fps=frames_per_second)
        clip.write_videofile(os.path.join(simulation_path, filename), codec='libx264')
        print(f"Simulation video saved!")
    
    env.close()

def parse_options():

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', default='antmaze-large-play-v2', help='Specify the environment name')
    parser.add_argument('--algo', choices=['cql', 'dt'], default='cql', help='Specify the algorithm you want to test')
    parser.add_argument('--ckpt_path', default=None, help='Insert the checkpoint path')
    parser.add_argument('--n_episodes', default=1, help='Number of episodes for each simulation')
    parser.add_argument('--simulation', choices=['True', 'False'], default='False', help='Generate simulation video')

    return parser.parse_args()

def run(*kwargs):
    opt = parse_options()
    main(opt)

if __name__ == '__main__':
    run()