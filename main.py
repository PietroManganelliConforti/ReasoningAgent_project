import random
import time
import argparse
import configparser
import os
from gym_sapientino_case.env import SapientinoCase
from gym_sapientino.core.configurations import (SapientinoAgentConfiguration,SapientinoConfiguration,)
from gym.wrappers import TimeLimit
from utils import get_config,get_colors
from models import  build_agent
from tensorforce.environments import Environment
from gym.wrappers import TimeLimit
from argparse import ArgumentParser
from trainer import Trainer
from argparse import ArgumentParser
from gym_sapientino import actions

def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description='Train an agent on SapientinoCase.')
    parser.add_argument('--config_file', default='config1.cfg')
    args = parser.parse_args()
    config_file = args.config_file

    configuration = configparser.ConfigParser()
    configuration.read(os.path.join('./configs/',config_file))
    tensorforce_config = configuration['TENSORFORCE']
    environment_config = get_config(configuration)

    #colors = get_colors(configuration['ENVIRONMENT']['reward_ldlf'])
    colors= ['red']    
    num_colors = len(colors)

    HIDDEN_STATE_SIZE = int(tensorforce_config['hidden_size'])
    NUM_STATES_AUTOMATON = num_colors+1 #a state for each color + the state in which we reach the goal. Is there also a state for the start? wip
    #SINK_ID=2  #the fail state has been removed
    AUTOMATON_STATE_ENCODING_SIZE = HIDDEN_STATE_SIZE*NUM_STATES_AUTOMATON
    NUM_EXPERTS = num_colors
    MAX_EPISODE_TIMESTEPS = int(tensorforce_config['max_timesteps'])

    AUTOMATON_STATE_ENCODING_SIZE = HIDDEN_STATE_SIZE*NUM_STATES_AUTOMATON

    env = SapientinoCase( 
        conf = environment_config, 
        reward_ldlf = configuration['ENVIRONMENT']['reward_ldlf'], 
        logdir = './experiments/'+configuration['OTHER']['name_dir_experiment']
        )

        # Limit the length of the episode
    environment = TimeLimit(env, MAX_EPISODE_TIMESTEPS)
    environment = Environment.create(environment =environment,max_episode_timesteps=MAX_EPISODE_TIMESTEPS,visualize =False)
    



#     #Dictionary containing the agent configuration parameters
    agent = build_agent("double_dqn",batch_size =int(tensorforce_config['batch_size']) ,
                        memory =int(tensorforce_config['memory']),
                        update_frequency=int(tensorforce_config['update_frequency']),
                        multi_step =int(tensorforce_config['multi_step']),
                        learning_rate=float(tensorforce_config['learning_rate']),
                        environment = environment,
                        num_states_automaton =NUM_STATES_AUTOMATON,
                        automaton_state_encoding_size=AUTOMATON_STATE_ENCODING_SIZE,

                        hidden_layer_size=HIDDEN_STATE_SIZE,

                        exploration =float(tensorforce_config['exploration']) ,

                        entropy_regularization=float(tensorforce_config['entropy_bonus']),)




    trainer = Trainer(agent,env,NUM_STATES_AUTOMATON,AUTOMATON_STATE_ENCODING_SIZE,
                                    num_colors=num_colors)

    EPISODES = int(tensorforce_config['episodes'])


#     #Train the agent

    training_results = trainer.train(episodes=EPISODES)

    print("Training of the agent complete: results are: ")
    print(training_results)
    

if __name__ == '__main__':
    main()

    

