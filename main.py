import argparse
import configparser
import os
from utils import get_colors, CustomEnv
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from trainer import Trainer
import json

def main(**kwargs):

    # Argument parsing
    parser = argparse.ArgumentParser(description='Train an agent on SapientinoCase.')
    parser.add_argument('--config_file', default='config1.cfg')
    args = parser.parse_args()
    config_file = args.config_file

    configuration = configparser.ConfigParser()
    configuration.read(os.path.join('./configs/',config_file))
    if kwargs:
        kwargs = kwargs['args']
        for conf in kwargs:
            if conf in configuration['TENSORFORCE'].keys():
                configuration['TENSORFORCE'][conf] = kwargs[conf]
            elif conf in configuration['ENVIRONMENT'].keys():
                configuration['ENVIRONMENT'][conf] = kwargs[conf]
            elif conf in configuration['AGENT'].keys():
                configuration['AGENT'][conf] = kwargs[conf]
            elif conf in configuration['OTHER'].keys():
                configuration['OTHER'][conf] = kwargs[conf]
            
    tensorforce_config = configuration['TENSORFORCE']
    env_config = configuration['ENVIRONMENT']
    colors = get_colors(env_config['reward_ldlf'])
    tg_reward = float(env_config['tg_reward'])
    num_colors = len(colors)
    NUM_EXPERTS = num_colors
    NUM_STATES_AUTOMATON = NUM_EXPERTS+1 
    HIDDEN_STATE_SIZE = int(tensorforce_config['hidden_size'])
    AUTOMATON_STATE_ENCODING_SIZE = HIDDEN_STATE_SIZE*NUM_STATES_AUTOMATON
    MAX_EPISODE_TIMESTEPS = int(tensorforce_config['max_timesteps'])
    EPISODES = int(tensorforce_config['episodes'])
    TG_REWARD = configuration['ENVIRONMENT']['tg_reward']
    DISCOUNT = float(tensorforce_config['discount'])
    if DISCOUNT > 0 and DISCOUNT < 1: 
        configuration['ENVIRONMENT']['reward_per_step'] = '0.0'

    # Limit the length of the episode
    environment = CustomEnv(configuration)
    environment = Environment.create(environment=environment,max_episode_timesteps=MAX_EPISODE_TIMESTEPS,visualize = True)
    
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
        learning_rate = dict(type='linear', unit='episodes', num_steps=EPISODES, #
                            initial_value=0.001, final_value=0.000005),
                        
        #learning_rate = float(tensorforce_config['learning_rate_initial_value']),

        #learning_rate=dict( type='exponential', unit='episodes', num_steps=1000,
        #                     initial_value=float(tensorforce_config['learning_rate_initial_value']), 
        #                     decay_rate=float(tensorforce_config['learning_rate_decay_value']),
        #                     min_value=0.000075),
        exploration = dict(type='linear', unit='episodes', num_steps=EPISODES, #WAS 5000
                            initial_value=0.75, final_value=0.001), 
        #exploration =dict( type='exponential', unit='episodes', num_steps=1,
        #                    initial_value=float(tensorforce_config['exploration_initial_value']), 
        #                    decay_rate=float(tensorforce_config['exploration_decay_value'])) ,
        saver=dict(directory='model'),
        summarizer=dict(directory='summaries',summaries=['reward','graph']),
        entropy_regularization = float(tensorforce_config['entropy_bonus']),
        discount = DISCOUNT,
        target_sync_frequency = int(tensorforce_config['target_sync_frequency']),
        target_update_weight = float(tensorforce_config['target_update_weights'])
        )



    trainer = Trainer(agent,environment,NUM_EXPERTS,AUTOMATON_STATE_ENCODING_SIZE, TG_REWARD, num_colors)
    training_results = trainer.train(episodes=EPISODES)

    print("Training of the agent complete: results are: ")
    print(training_results)
    dict_res = training_results
    dict_res['configuration'] = dict(configuration._sections)
    return dict_res

if __name__ == '__main__':
    main()
    # import time
    # var_cycle_on = 'exploration'
    # to_cycle = [ 3.0, 2.0 , 1.0]
    # data_to_write = {}
    # for num, x in enumerate(to_cycle):
    #     print(f"[INFO] testing {var_cycle_on} with value {to_cycle[num]}")
    #     dict_result = main(args={var_cycle_on: str(x)})
    #     data_to_write[x] = dict_result
    # time.sleep(60)
    # with open('training_results.json', 'r+') as outfile:
    #     data = json.load(outfile)
    #     out = {var_cycle_on: data_to_write}
    #     data.update(out)
    #     outfile.seek(0)
    #     json.dump(data, outfile)

    

