import os
import numpy as np

from gym.spaces import Tuple, Box
from gym import ObservationWrapper
from gym_sapientino_case.env import SapientinoCase
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)
from gym_sapientino.core.actions import ContinuousCommand

def one_hot_encode(x,size, num_labels):
    ret = np.zeros(size,dtype = np.float)
    if size%num_labels !=0:
        return ret
    else:
        block_size = int(size/num_labels)
        i = 0
        k = 0
        while(i<size):
            if k == x:
                ret[i:i+block_size] = 1.0
            k+=1
            i = i+block_size
        return ret

def get_agent_configuration(config):
    initial_position_x = float(config['initial_position_x'])
    initial_position_y = float(config['initial_position_y'])
    angular_speed = float(config['angular_acceleration'])
    acceleration = float(config['acceleration'])
    max_velocity = float(config['max_velocity'])
    min_velocity = float(config['min_velocity'])
    return SapientinoAgentConfiguration(
                initial_position= (initial_position_x, initial_position_y),
                commands=ContinuousCommand,
                angular_speed=angular_speed,
                acceleration=acceleration,
                max_velocity=max_velocity,
                min_velocity=min_velocity,
            )

def get_env_configuration(env_config, agent_config):
    grid_map_file = os.path.join('./maps/', env_config['map_file'])
    with open(grid_map_file) as f:
        grid_map = f.readlines()
        grid_map = ''.join(grid_map)
    reward_outside_grid = float(env_config['reward_outside_grid'])
    reward_duplicate_beep = float(env_config['reward_duplicate_beep'])
    reward_per_step = float(env_config['reward_per_step'])
    return SapientinoConfiguration(
        agent_configs=(agent_config,),
        grid_map=grid_map,
        reward_outside_grid=reward_outside_grid,
        reward_duplicate_beep=reward_duplicate_beep,
        reward_per_step=reward_per_step,
    )

def get_config(config):
    env_config = config['ENVIRONMENT']
    agent_config = get_agent_configuration(config['AGENT'])
    configuration = get_env_configuration(env_config, agent_config)
    return configuration

def get_colors(reward_ldlf):
    import re
    pattern_colors_on_ldlf = re.compile(r'\![a-z]+\*',re.IGNORECASE)
    colors = [ re.sub('\!|\*', '', x) for x in pattern_colors_on_ldlf.findall(reward_ldlf)]
    return colors

def test_environment(env):
    import time
    import random
    while True:
        # Init episode
        obs = env.reset()
        done = False
        cum_reward = 0.0
        print(f"\n> Env reset.\nInitial observation {obs}")
        while not done:
            # Render
            env.render()
            # Compute action
            action = random.randint(0, env.action_space.n - 1)
            # Move env
            obs, reward, done, _ = env.step(action)
            cum_reward += reward
            print("Step.", f"Action {action}", f"Observation {obs}", f"Reward {reward}", f"Done {done}", sep="\n  ",)
            # Let us see the screen
            time.sleep(0.1)

class CustomEnv(ObservationWrapper):
    def __init__(self, configuration):
        environment_config = get_config(configuration)
        self.num_of_experts = len(get_colors(configuration['ENVIRONMENT']['reward_ldlf']))
        self.automaton_state_encoding_size = int(configuration['TENSORFORCE']['hidden_size'])*(self.num_of_experts+1)
        env = SapientinoCase( 
            conf = environment_config, 
            reward_ldlf = configuration['ENVIRONMENT']['reward_ldlf'], 
            logdir = './experiments/'+configuration['OTHER']['name_dir_experiment']
            )
        ObservationWrapper.__init__(self,env)
        self.observation_space = Tuple((env.observation_space[0], 
            Box(low=np.array([0]*self.automaton_state_encoding_size), high=np.array([1]*self.automaton_state_encoding_size), dtype=np.float32)))

    def observation(self, observation):
        return (observation[0], self.encode(observation[1][0]))
    
    def encode(self, automaton_state):
            """
                Prepare the encoded automaton state.
            """
            one_hot_encoding = one_hot_encode(automaton_state, self.automaton_state_encoding_size, self.num_of_experts)
            one_hot_encoding = one_hot_encoding.astype(np.float32)

            return one_hot_encoding