import random
import time
# import argparse
# import configparser
import os
from gym_sapientino_case.env import SapientinoCase
from gym_sapientino.core.configurations import (SapientinoAgentConfiguration,SapientinoConfiguration,)
from gym.wrappers import TimeLimit
from utils import get_config
from models import  build_agent
from tensorforce.environments import Environment
from gym.wrappers import TimeLimit
from argparse import ArgumentParser
from trainer import Trainer
from argparse import ArgumentParser
from gym_sapientino import actions

def main():

 #Handle command line arguments
    # parser = argparse.ArgumentParser()

    # parser.add_argument('--batch_size', type = int, default = 64,help= 'Experience batch size.')
    # parser.add_argument('--memory', type = int, default = None,help= 'Memory buffer size. Used by agents that train with replay buffer.')
    # parser.add_argument('--multi_step',type = int, default = 10, help="Agent update optimization steps.")
    # parser.add_argument('--update_frequency', type = int, default = None, help="Frequency of the policy updates. Default equals to batch_size.")
    # parser.add_argument('--num_colors', type = int, default = 1, help="Number of distinct colors in the map.")
    # parser.add_argument('--learning_rate', type = float, default = 0.001, help="Learning rate for the optimization algorithm")
    # parser.add_argument('--exploration', type = float, default = 0.0, help = "Exploration for the epsilon greedy algorithm.")
    # parser.add_argument('--entropy_bonus', type = float, default = 0.0, help ="Entropy bonus for the 'extended' loss of PPO. It discourages the policy distribution from being “too certain” (default: no entropy regularization." )
    # parser.add_argument('--hidden_size', type = int, default = 64, help="Number of neurons of the hidden layers of the network.")
    # parser.add_argument('--max_timesteps', type = int, default = 300, help= "Maximum number of timesteps each episode.")
    # parser.add_argument('--episodes', type = int, default = 1000, help = "Number of training episodes.")
    # parser.add_argument('--path', type = str, default = None, help = "Path to the map file inside the file system.")
    # parser.add_argument("--sequence", nargs="+", default=None, help="Goal sequence for the training specified as a list of strings.")



    # args = parser.parse_args()

    #Collect some information from the argument parser.
    batch_size = 64
    memory = 300
    multi_step = 10
    update_frequency = 64 #default value equal to batch size
    num_colors = 1
    learning_rate = 0.001
    exploration = 0.0
    entropy_bonus = 0.0
    hidden_size = 64
    max_timesteps = 300
    episodes = 1000
    path = None
    sequence = ["red"]

    parameters = [batch_size,memory,multi_step,update_frequency,num_colors,learning_rate,
    learning_rate,exploration,entropy_bonus,hidden_size,max_timesteps,episodes,path,sequence]
    
    #Extract the map from the command line arguments
    if not path:
        if num_colors == 1:
            map_file = os.path.join('.','maps/map1.txt')
        elif num_colors == 2:
            map_file = os.path.join('.', 'maps/map2.txt')
        # elif num_colors == 4:
        #     map_file = os.path.join('.','maps/map4_easy.txt')
        else:
            raise AttributeError('Map with ', num_colors,' colors not supported by default. Specify a path for a map file.')
    else:
        map_file = path



    #Extract the goal sequence form the command line arguments
    if not sequence:
        if num_colors == 1:
            colors = ['red']
        elif num_colors == 2:
            colors = ['red','yellow']
        # elif num_colors == 4:
        #     colors = ['blue','red','yellow','green']
        else:
            raise AttributeError('Map with ', num_colors,' colors not supported by default. Specify a path for a map file.')
    else:
        colors = sequence


    #Log directory for the automaton states.
    log_dir = os.path.join('.','log_dir')


    HIDDEN_STATE_SIZE = hidden_size
    #I dont understand how to parse effectively the txt map_file. Probably it needs a whole function
    #to obtain such format from the map_file path. It's easier and faster to directly write the map 
    #here imho. Not so elegant, but time efficient
    _map = """\
    |           |
    |           |
    |           |
    |     #    r|
    |     #     |
    |###########|"""

    #not working here!! WIP
    #setting parameters to be given to sapientinoCase
    agent_conf = SapientinoAgentConfiguration(
                initial_position=(2, 2),
                commands=actions.ContinuousCommand,
                angular_speed=30.0,
                acceleration=0.10,
                max_velocity=0.40,
                min_velocity=0.0,
            )

    conf = SapientinoConfiguration(
        agent_configs=(agent_conf,),
        grid_map=_map,
        reward_outside_grid=0.0,
        reward_duplicate_beep=0.0,
        reward_per_step=0.0,
    )


    environment = SapientinoCase(
        conf=conf,

        # map_file = map_file,
        logdir =log_dir
        )


  #Default tensorforce update frequency is batch size.
    if not update_frequency:
        update_frequency = batch_size

    #Default dqn memory.
    if not memory:
        memory = 500


    #the fail state has been removed
    NUM_STATES_AUTOMATON = num_colors+1 #a state for each color + the state in which we reach the goal. Is there also a state for the start? wip
    #SINK_ID=2
    AUTOMATON_STATE_ENCODING_SIZE = HIDDEN_STATE_SIZE*NUM_STATES_AUTOMATON
    NUM_EXPERTS = num_colors
    MAX_EPISODE_TIMESTEPS = max_timesteps

    environment = TimeLimit(environment, MAX_EPISODE_TIMESTEPS)
    environment = Environment.create(environment =environment,max_episode_timesteps=MAX_EPISODE_TIMESTEPS,visualize =False)
    
    AUTOMATON_STATE_ENCODING_SIZE = HIDDEN_STATE_SIZE*NUM_STATES_AUTOMATON

    #Dictionary containing the agent configuration parameters
    agent = build_agent("double_dqn",batch_size = batch_size,
                        memory =memory,
                        update_frequency=update_frequency,
                        multi_step = multi_step,
                        learning_rate=learning_rate,

                        environment = environment,
                        num_states_automaton =NUM_STATES_AUTOMATON,
                        automaton_state_encoding_size=AUTOMATON_STATE_ENCODING_SIZE,

                        hidden_layer_size=HIDDEN_STATE_SIZE,

                        exploration =exploration,

                        entropy_regularization=entropy_bonus,)


    #Debugging prints
    DebugBool=True
    if(DebugBool):
        print("Istantiated an agent for training with parameters: ")
        print(parameters)

        print("The goal sequence is: ")
        print(colors)

    trainer = Trainer(agent,environment,NUM_STATES_AUTOMATON,AUTOMATON_STATE_ENCODING_SIZE,
                                    num_colors=num_colors)

    EPISODES = episodes


    #Train the agent

    training_results = trainer.train(episodes=EPISODES)

    print("Training of the agent complete: results are: ")
    print(training_results)
    

if __name__ == '__main__':
    main()

    

