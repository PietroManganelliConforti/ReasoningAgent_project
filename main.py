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
    


    #Dictionary containing the agent configuration parameters
    #Other parameters
    #non_markovian: (bool) boolean flag specifying whether or not to istantiate an agent with a non markovian policy network. In the project the markovian agent is used essentially as a baseline for comparisons.
    #saver: (dict)

    agent = build_agent(agent= "double_dqn",                                            #agent: (string) the name of the deep reinforcement learning algorithm used to train the agent.
                        batch_size =int(tensorforce_config['batch_size']) ,             #batch_size: (int) the size of experience batch collected by the agent.
                        memory =int(tensorforce_config['memory']),                      #memory: (int) the size of the agent memory.
                        update_frequency=int(tensorforce_config['update_frequency']),   #update_frequency: (int) frequency of updates (default 20).
                        multi_step =int(tensorforce_config['multi_step']),              #multi_step: (int) number of optimization steps, update_frequency * multi_step should be at least 1 if relative subsampling_fraction (default: 10).
                        learning_rate=float(tensorforce_config['learning_rate']),       #learning_rate: (float) optimizer learning rate (default: 0.001)
                        environment = environment,                                      #environment: (tensorforce.environments.Environment) istance of the tensorforce environment in which is trained.
                        num_states_automaton =NUM_STATES_AUTOMATON,                     #num_states_automaton: (int) number of states of the goal state DFA.                     
                        automaton_state_encoding_size=AUTOMATON_STATE_ENCODING_SIZE,    #automaton_state_encoding_size: (int) size of the binary encoding of the automaton state. See the report in report/pdf in section "Non markovian agent" for further details.
                        hidden_layer_size=HIDDEN_STATE_SIZE,                            #hidden_layer_size: (int) number of neurons of the policy network hidden layer (default implementation features two hidden layers with an equal number of neurons).
                        exploration =float(tensorforce_config['exploration']) ,         #exploration: (float) exploration, defined as the probability for uniformly random output in case of bool and int actions, and the standard deviation of Gaussian noise added to every output in case of float actions, specified globally or per action-type or -name (default: no exploration).
                        entropy_regularization=float(tensorforce_config['entropy_bonus']),# entropy_regularization: (float) entropy regularization loss weight, to discourage the policy distribution from being “too certain” (default: no entropy regularization).
                        )


    trainer = Trainer(agent,env,NUM_STATES_AUTOMATON,AUTOMATON_STATE_ENCODING_SIZE,
                                    num_colors=num_colors)

    EPISODES = int(tensorforce_config['episodes'])


#     #Train the agent

    training_results = trainer.train(episodes=EPISODES)

    print("Training of the agent complete: results are: ")
    print(training_results)
    

if __name__ == '__main__':
    main()

    

