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
from tensorforce.agents import Agent
from gym import ObservationWrapper
from gym.wrappers import TimeLimit
from trainer import Trainer
from gym_sapientino import actions
from gym.spaces import Tuple, Box, MultiDiscrete
from one_hot import one_hot_encode
import numpy as np

class CustomEnv(ObservationWrapper):
    def __init__(self, configuration):
        environment_config = get_config(configuration)
        tensorforce_config = configuration['TENSORFORCE']
        self.num_of_experts = len(get_colors(configuration['ENVIRONMENT']['reward_ldlf']))
        self.automaton_state_encoding_size = int(tensorforce_config['hidden_size'])*(self.num_of_experts+1)
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


def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description='Train an agent on SapientinoCase.')
    parser.add_argument('--config_file', default='config1.cfg')
    args = parser.parse_args()
    config_file = args.config_file

    configuration = configparser.ConfigParser()
    configuration.read(os.path.join('./configs/',config_file))
    tensorforce_config = configuration['TENSORFORCE']

    colors = get_colors(configuration['ENVIRONMENT']['reward_ldlf'])
    num_colors = len(colors)

    HIDDEN_STATE_SIZE = int(tensorforce_config['hidden_size'])
    NUM_STATES_AUTOMATON = num_colors+1 
    AUTOMATON_STATE_ENCODING_SIZE = HIDDEN_STATE_SIZE*NUM_STATES_AUTOMATON
    NUM_EXPERTS = num_colors
    MAX_EPISODE_TIMESTEPS = int(tensorforce_config['max_timesteps'])

    # Limit the length of the episode
    environment = CustomEnv(configuration)
    environment = Environment.create(environment=environment,max_episode_timesteps=MAX_EPISODE_TIMESTEPS,visualize =True)
    
    agent = Agent.create(
        agent='double_dqn', 
        environment=environment, 
        memory=int(tensorforce_config['memory']), 
        batch_size =int(tensorforce_config['batch_size']),
        network=dict(type = 'custom',
                        layers= [
                            dict(type = 'retrieve',tensors= ['gymtpl0']),
                            dict(type = 'linear_normalization'),
                            dict(type='dense', bias = True,activation = 'tanh',size=AUTOMATON_STATE_ENCODING_SIZE),
                            dict(type= 'register',tensor = 'gymtpl0-dense1'),

                            #Perform the product between the one hot encoding of the automaton and the output of the dense layer.
                            dict(type = 'retrieve',tensors=['gymtpl0-dense1','gymtpl1'], aggregation = 'product'),
                            dict(type='dense', bias = True,activation = 'tanh',size=AUTOMATON_STATE_ENCODING_SIZE),
                            dict(type= 'register',tensor = 'gymtpl0-dense2'),
                            dict(type = 'retrieve',tensors=['gymtpl0-dense2','gymtpl1'], aggregation = 'product'),
                            dict(type='register',tensor = 'gymtpl0-embeddings'),

                        ],

                        ),
        update_frequency=int(tensorforce_config['update_frequency']),
        learning_rate=float(tensorforce_config['learning_rate']),
        exploration =float(tensorforce_config['exploration']) ,
        saver=dict(directory='model'),
        summarizer=dict(directory='summaries',summaries=['reward','graph']),
        entropy_regularization = float(tensorforce_config['entropy_bonus'])
        )

<<<<<<< HEAD
    # # Train for 100 episodes
    # for episode in range(100):

    #     # Episode using act and observe
    #     states = environment.reset()
    #     terminal = False
    #     sum_rewards = 0.0
    #     num_updates = 0
    #     while not terminal:
    #         actions = agent.act(states=states)
    #         states, terminal, reward = environment.execute(actions=actions)
    #         num_updates += agent.observe(terminal=terminal, reward=reward)
    #         sum_rewards += reward
    #     print('Episode {}: return={} updates={}'.format(episode, sum_rewards, num_updates))

    # # Evaluate for 100 episodes
    # sum_rewards = 0.0
    # for _ in range(100):
    #     states = environment.reset()
    #     internals = agent.initial_internals()
    #     terminal = False
    #     while not terminal:
    #         actions, internals = agent.act(
    #             states=states, internals=internals, independent=True, deterministic=True
    #         )
    #         states, terminal, reward = environment.execute(actions=actions)
    #         sum_rewards += reward
    # print('Mean evaluation return:', sum_rewards / 100.0)

    # # Close agent and environment
    # agent.close()
    # environment.close()

    trainer = Trainer(agent,environment,NUM_STATES_AUTOMATON,AUTOMATON_STATE_ENCODING_SIZE,
=======

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
>>>>>>> 152a81488ba60ad8b1e89dae43920f4fee6180c5
                                    num_colors=num_colors)

    EPISODES = int(tensorforce_config['episodes'])


#     #Train the agent

    training_results = trainer.train(episodes=EPISODES)

    print("Training of the agent complete: results are: ")
    print(training_results)
    

if __name__ == '__main__':
    main()

    

