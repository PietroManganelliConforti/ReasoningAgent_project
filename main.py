import argparse
import configparser
import os
from utils import get_colors, CustomEnv
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from trainer import Trainer

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
    NUM_EXPERTS = num_colors
    NUM_STATES_AUTOMATON = NUM_EXPERTS+1 

    HIDDEN_STATE_SIZE = int(tensorforce_config['hidden_size'])
    AUTOMATON_STATE_ENCODING_SIZE = HIDDEN_STATE_SIZE*NUM_STATES_AUTOMATON
    MAX_EPISODE_TIMESTEPS = int(tensorforce_config['max_timesteps'])
    EPISODES = int(tensorforce_config['episodes'])

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

    trainer = Trainer(agent,environment,NUM_STATES_AUTOMATON,AUTOMATON_STATE_ENCODING_SIZE,num_colors=num_colors)
    training_results = trainer.train(episodes=EPISODES)

    print("Training of the agent complete: results are: ")
    print(training_results)
    

if __name__ == '__main__':
    main()

    

