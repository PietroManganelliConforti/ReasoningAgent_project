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

    trainer = Trainer(agent,environment,NUM_STATES_AUTOMATON,AUTOMATON_STATE_ENCODING_SIZE,num_colors=num_colors)

    EPISODES = int(tensorforce_config['episodes'])


#     #Train the agent

    training_results = trainer.train(episodes=EPISODES)

    print("Training of the agent complete: results are: ")
    print(training_results)
    

if __name__ == '__main__':
    main()

    

