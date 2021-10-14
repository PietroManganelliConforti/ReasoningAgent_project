from enum import auto
from tensorforce.agents import Agent
from tensorforce.environments import Environment
from tqdm.auto import tqdm
from utils import get_automaton_state_from_encoding
import numpy as np


DEBUG = False


class Trainer(object):
    def __init__(self,agent,environment,number_of_experts,
                 automaton_encoding_size, num_colors = 1,):

        self.number_of_experts = number_of_experts
        self.automaton_encoding_size = automaton_encoding_size
        self.agent = agent
        self.environment = environment
        self.num_colors = num_colors

    def get_reward_from_automaton_state(self, current_automaton_state, previous_automaton_state, terminal):
        reward = 0.0
        for i in range(1, self.num_colors+1):
            if current_automaton_state == i and previous_automaton_state == i-1:
                reward = 500.0
                if current_automaton_state == self.num_colors:
                    terminal = True
                return reward, terminal
        return reward, terminal

    def train(self,episodes = 1000):
        num_time_visited_goal = 0
        cum_reward = 0.0
        agent = self.agent
        environment = self.environment
        pbar = tqdm(range(episodes),desc='[Training]',leave = True)
        try:
            for episode in pbar:
                terminal = False

                #I obtain the obs and the automaton state to begin with
                states = environment.reset()

                #automaton_state = get_automaton_state_from_encoding(states['gymtpl1'], self.number_of_experts, self.automaton_encoding_size)
                automaton_state = environment._environment.environment.aut_state_obs

                #I set the initial parameters to launch the training
                prevAutState = 0
                #Save the reward that you reach in the episode inside a linked list. This will be used for nice plots in the report.
                ep_reward = 0.0

                while not terminal:
                    #I start the training setting the actions
                    actions = agent.act(states=states)

                    #I execute(?) the environment obtaining the states, the reward and if Im in a terminal condition or not
                    states, terminal, reward = environment.execute(actions=actions)
                    
                    #Extract gym sapientino state and the state of the automaton.
                    #automaton_state = get_automaton_state_from_encoding(states['gymtpl1'], self.number_of_experts, self.automaton_encoding_size)
                    
                    automaton_state = environment._environment.environment.aut_state_obs
                    
                    """
                        Reward shaping.
                    """
                    reward, terminal = self.get_reward_from_automaton_state(automaton_state, prevAutState, terminal)
                    
                   

                    #I update the previous state with the state in which I was in this training cycle,regardless of the fact
                    #that I have transitated in a new relevant state.
                    prevAutState = automaton_state


                    #Update the cumulative reward during the training.
                    cum_reward += reward

                    #Update the episode reward during the training
                    ep_reward += reward
                    if terminal == True: num_time_visited_goal += 1
                    pbar.set_postfix({'current_reward': reward, 
                                      'episode_reward': ep_reward, 
                                      'total_reward': cum_reward, 
                                      'visited_goal_for_n_time': num_time_visited_goal})

                    #let the automaton observe the reward obtained with the last action, and if he completed the task
                    agent.observe(terminal=terminal, reward=reward)
                    if terminal == True:
                        states = environment.reset()



            #Close both the agent and the environment.
            self.agent.close()
            self.environment.close()


            return dict(cumulative_reward_nodiscount = cum_reward,
                        average_reward_nodiscount = cum_reward/episodes)
        finally:

           #Let the user interrupt
           pass

