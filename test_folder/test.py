from msgpack_numpy import encode
from tensorforce import Agent, Environment
from gym_sapientino_case.env import SapientinoCase
from gym_sapientino.core.configurations import (SapientinoAgentConfiguration,SapientinoConfiguration,)
from gym.wrappers import TimeLimit
from gym_sapientino.core.actions import Command, GridCommand
from gym_sapientino.core.actions import ContinuousCommand
import numpy as np

def main():


    agent_config = SapientinoAgentConfiguration(
                initial_position= (0.0, 0.0),
                commands=ContinuousCommand,
                angular_speed=10.0,
                acceleration=0.2,
                max_velocity=0.0,
                min_velocity=0.4,
            )

    config = SapientinoConfiguration(
        agent_configs=(agent_config,),
        reward_outside_grid = 0.0,
        reward_duplicate_beep= 0.0,
        reward_per_step= 0.0,
    )

    print()
    print(config)
    print()


    env = SapientinoCase( 
        conf = config, 
        reward_ldlf = '<!red*; red>end', 
        logdir = "test_folder/experiments2"
        )


    environmentTL = TimeLimit(env, 500)
    environment = Environment.create(environment= environmentTL, max_episode_timesteps= 500)
    
    environment1 = Environment.create(environment='gym', level='CartPole', max_episode_timesteps=500)

    agent1 = Agent.create(agent='double_dqn',environment=environment ,batch_size=64, memory= 600)
    
    agent = Agent.create(
        
            agent='double_dqn',
            environment=environment , 
            batch_size=64, 
            memory= 600,
        
            states = dict(
                gymtpl0 = dict(type = 'float',shape= (7,),min_value = -np.inf,max_value = np.inf),
                gymtpl1 = dict(type ='int',shape=(1,))
                ),

            network=dict(
                        type = 'custom',
                        layers= [
                        dict(type = 'retrieve',tensors= ['gymtpl0']),
                        dict(type = 'linear_normalization'),
                        dict(type='dense', bias = True,activation = 'tanh',size=64), #64 hidden layer size

                        #Perform the product between the one hot encoding of the automaton and the output of the dense layer.
                        dict(type='dense', bias = True,activation = 'tanh',size=64), #64 hidden layer size
                        dict(type='register',tensor = 'gymtpl0-embeddings'), ],
                        )     
        )


    states = environment.reset()
    #print(states["gymtpl0"])
    actions = agent.act(states)
    #print(actions)
    states, terminal, reward = environment.execute(actions=actions)
    num_updates = agent.observe(terminal=terminal, reward=reward)
    #print(num_updates)





    

    # Train for 100 episodes
    for episode in range(100):

        # Episode using act and observe
        states = environment.reset()
        #states_pole = environment_pole.reset()

        terminal = False
        sum_rewards = 0.0
        num_updates = 0
        while not terminal:
            actions = agent.act(states=states["gymtpl0"])
            states, terminal, reward = environment.execute(actions=actions)
            num_updates += agent.observe(terminal=terminal, reward=reward)
            sum_rewards += reward
        print('Episode {}: return={} updates={}'.format(episode, sum_rewards, num_updates))

    # Evaluate for 100 episodes
    sum_rewards = 0.0
    for i in range(100):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent.act(
                states=states, internals=internals, independent=True, deterministic=True
            )
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward
        print('Episode {}'.format(i))
    print('Mean evaluation return:', sum_rewards / 100.0)


    

    # Close agent and environment
    agent.close()
    environment.close()


if __name__ == '__main__':
    main()