import minari 
import d4rl
import gym
import h5py 


def main():

    env_name = "Walker2d-v2"
    dataset = minari.load_dataset(f'/home/server/Desktop/sebastiano/offline_rl/assets/dataset/{env_name}')

    print('*'*50)
    print('Observation space: ', dataset.observation_space)
    print('State dim: ', dataset.observation_space.shape[0])
    print('Action space ', dataset.action_space)
    print('Action dim: ', dataset.action_space.shape[0])
    print('Total episodes ', dataset.total_episodes)
    print('Total steps ', dataset.total_steps) 
    print('*'*50)

    states = []
    actions = []
    rewards = [] 
    next_states = []
    terminations = []
    truncations = []

    for episode in dataset.iterate_episodes():
        observations = episode.observations
        actions.extend(episode.actions)
        rewards.extend(episode.rewards)
        terminations.extend(episode.terminations)
        truncations.extend(episode.truncations)
       
        states.extend(observations[:-1])
        next_states.extend(observations[1:])

    assert len(states) == len(next_states)

    print("First few states:", states[:5])
    print("First few actions:", actions[:5])
    print("First few rewards:", rewards[:5])
    print("First few next_states:", next_states[:5])
    print("First few terminations:", terminations[:5])
    print("First few truncations:", truncations[:5])



if __name__ == '__main__':
    main()