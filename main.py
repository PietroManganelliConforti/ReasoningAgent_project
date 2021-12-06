import argparse
import configparser
import os
import shutil
from utils import get_colors, CustomEnv
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from runner import Runner

import warnings
warnings.filterwarnings("ignore")

def main():

    parser = argparse.ArgumentParser(description='Train an agent on SapientinoCase.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config_file')
    group.add_argument('--trained_model_path')
    args = parser.parse_args()

    #--------------------------------
    #       LOAD CONFIGURATION      
    #--------------------------------
    load_agent = True if args.trained_model_path is not None else False
    train_agent = not load_agent
    configuration = configparser.ConfigParser()
    if not load_agent:
        config_file = args.config_file
        configuration.read(os.path.join('./configs/',config_file))
    else:
        configuration.read(os.path.join(args.trained_model_path, 'config.cfg'))
    tensorforce_config = configuration['TENSORFORCE']
    env_config = configuration['ENVIRONMENT']
    runner_config = configuration['RUNNER']



    #--------------------------------
    #       CREATE ENVIRONMENT      
    #--------------------------------
    MAX_EPISODE_TIMESTEPS = int(env_config['max_timesteps'])
    colors= get_colors(env_config['reward_ldlf'])
    NUM_EXPERTS = len(colors)
    NUM_STATES_AUTOMATON = NUM_EXPERTS+1 
    TG_REWARD = float(env_config['tg_reward'])
    HIDDEN_STATE_SIZE = int(tensorforce_config['hidden_size'])
    AUTOMATON_STATE_ENCODING_SIZE = HIDDEN_STATE_SIZE*NUM_STATES_AUTOMATON
    goal_reward_reduction_rate = float(runner_config['goal_reward_reduction_rate'])

    customEnvironment = CustomEnv(configuration)
    environment = Environment.create(environment=customEnvironment,max_episode_timesteps=MAX_EPISODE_TIMESTEPS,visualize = True)

    if not load_agent:

        #--------------------------------
        #          CREATE AGENT         
        #--------------------------------
        AGENT_TYPE = configuration['AGENT']['algorithm'].lower()
        EPISODES = int(runner_config['episodes'])
        DISCOUNT = float(tensorforce_config['discount'])
        LR_INIT = float(tensorforce_config['learning_rate_initial_value'])
        LR_FINAL = float(tensorforce_config['learning_rate_final_value'])
        EXP_INIT = float(tensorforce_config['exploration_initial_value'])
        EXP_FINAL = float(tensorforce_config['exploration_final_value'])

        saved_experiments = [folder for folder in os.listdir('./model/') if AGENT_TYPE in folder]
        save_folder = './model/'+AGENT_TYPE+'_'+str(len(saved_experiments))

        if DISCOUNT > 0 and DISCOUNT < 1: 
            configuration['ENVIRONMENT']['reward_per_step'] = '0.0'
        if AGENT_TYPE == 'ddqn': AGENT_TYPE = 'double_dqn'

        args_for_agent = {
            'batch_size':int(tensorforce_config['batch_size']),
            'network':dict(type = 'custom',
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
            'update_frequency':int(tensorforce_config['update_frequency']),
            'learning_rate': dict(type='linear', unit='episodes', num_steps=EPISODES,
                                initial_value= LR_INIT, final_value=LR_FINAL),
            'exploration': dict(type='linear', unit='episodes', num_steps=EPISODES,
                                initial_value=EXP_INIT, final_value=EXP_FINAL), 
            'summarizer':dict(directory='summaries',summaries=['reward','graph']),
            'entropy_regularization': float(tensorforce_config['entropy_bonus']),
            'discount': DISCOUNT
            }

        if AGENT_TYPE == 'double_dqn':
            args_for_agent['memory'] = int(tensorforce_config['memory'])
            args_for_agent['target_sync_frequency'] = int(tensorforce_config['target_sync_frequency'])
            args_for_agent['target_update_weight'] = float(tensorforce_config['target_update_weights'])

        if AGENT_TYPE == 'ppo': 
            args_for_agent['memory'] = 'minimum'

        agent = Agent.create(agent=AGENT_TYPE, environment=environment, **args_for_agent)        

    else: 

        #--------------------------------
        #           LOAD AGENT          
        #--------------------------------
        agent = Agent.load(args.trained_model_path)

    #--------------------------------
    #         CREATE RUNNER         
    #--------------------------------
    runner = Runner(agent,environment,NUM_EXPERTS,AUTOMATON_STATE_ENCODING_SIZE, TG_REWARD, goal_reward_reduction_rate)

    if train_agent:

        #--------------------------------
        #          TRAIN AGENT          
        #--------------------------------
        training_results = runner.train(episodes=EPISODES)
        if training_results != None: 
            print("Training of the agent complete.\n The results are: ", training_results)

        #--------------------------------
        #          SAVE AGENT           
        #--------------------------------
        runner.agent.save(save_folder)
        shutil.copy(os.path.join('./configs/',config_file), os.path.join(save_folder, 'config.cfg'))
    else:

        #--------------------------------
        #         EVALUATE AGENT        
        #--------------------------------
        mean_evaluation_reward = runner.evaluate(episodes=100)
        if mean_evaluation_reward != None:
            print("Evaluation of the agent complete.\n Mean evaluation result is: ", mean_evaluation_reward)

    runner.close()
    return 

if __name__ == '__main__':
    main()

    

