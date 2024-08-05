# Mujoco installation pipeline

1) **create venv**: 
```bash
python3.10 -m venv <env_name>
```
or create conda environment:
```bash
conda create --name <env_name>
```

2) **Download mujoco binaries for Linux**: https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
3) **Extract and move folder**:
```bash
mkdir ~/.mujoco
cd PATH_TO_EXTRACTED_FOLDER
mv mujoco210 ~/.mujoco/mujoco210
```
4) **Export mujoco path variable in bashrc**:
```bash
gedit ~/.bashrc
## write the next line at the end of .bashrc file
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nitishgupta/.mujoco/mujoco210/bin
```
5) **Install cython**: pip install 'cython<3'
```bash
pip install 'cython<3'
```

## If you encounter the "osmesa.h" fatal error:
- If you have sudo privileges:
```bash
sudo apt-get install libosmesa6-dev
```
- If you don't have sudo privileges:
```bash
$ conda activate mujoco_env
$ conda install -c conda-forge glew
$ conda install -c conda-forge mesalib
$ conda install -c anaconda mesa-libgl-cos6-x86_64
$ conda install -c menpo glfw3
```

## If you encounter the "patchelf" file not found error:
- If you have sudo privileges:
```bash
sudo apt-get install patchelf
```
- In both cases you can do:
```bash
pip install patchelf
```

## Test the installs to see if everything works:
```python 
import mujoco_py
# running build_ext
import os
mj_path = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
print(sim.data.qpos)
sim.step()
print(sim.data.qpos)
```

# D4RL installation 
Official GitHub page: https://github.com/Farama-Foundation/D4RL.
1) Try:
```bash 
git clone https://github.com/Farama-Foundation/d4rl.git
cd d4rl
pip install -e .
```
2) Alternatively:
```bash 
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

## Test the installs to see if everything works:
If the next code works fine everything should be all right, since d4rl requires mujoco to be installed and working in order to be executed.
The code should produce a dataset based on the chosen environment and save it in **/home/server/.d4rl/datasets**.


```python 
import gym
import d4rl # Import required to register environments, you may need to also import the submodule

# Create the environment
env = gym.make('maze2d-umaze-v1')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations']) # An N x dim_observation Numpy array of observations

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
dataset = d4rl.qlearning_dataset(env)
```

Eventually you might wanna do:
```bash
pip install gym==0.21.0
pip install --upgrade d4rl
```

### NB (taken directly from the d4rl github page): 
Datasets are automatically downloaded to the ~/.d4rl/datasets directory when get_dataset() is called. If you would like to change the location of this directory, you can set the $D4RL_DATASET_DIR environment variable to the directory of your choosing, or pass in the dataset filepath directly into the get_dataset method.

## USAGE EXAMPLE - REPRODUCING RESULTS - CQL
Official CORL GitHub page: https://github.com/corl-team/CORL?tab=readme-ov-file#gym-mujoco.

If you want to reproduce the results obtained in https://github.com/corl-team/CORL?tab=readme-ov-file#gym-mujoco:
1) Choose an environment 
2) Copy and paste the **<configuration_file>.yaml** of the environment in the **configs** folder
3) If you want to save the checkpoints add the line:
```
checkpoints_path: <path_to_dir>
```
at the end of the **configuration.yaml** file

4) To train the algorithm with the chosen configuration:
```bash
python algorithms/training/cql.py --config=<path_to_config.yaml> [optional: --epochs=<n_epochs>]
```
5) This code uses **wandb** to save the statistics, if you have a wandb account and you want to see the live statistics modify the next line of code in the **cql.py** file:
```python
wandb.login(key="<your wandb API key>", force=True)

```

if you don't have a wandb account, comment all the lines of code that use wandb in the **cql.py** file.

## USAGE EXAMPLE - TESTING - CQL
To test the trained algorithm you can use the **algorithms_testing.py** script in the **scripts** folder:
```bash
python scripts --env_name <specify the env name> --algo <cql, dt> --ckpt_path <path to the .pt file that you want to test> --n_episodes <number of episodes, default=1> --simulation <True/False to generate a video simulation of the environment and agent>
```
