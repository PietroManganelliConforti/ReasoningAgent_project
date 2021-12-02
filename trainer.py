from tqdm.auto import tqdm
import numpy as np

class Trainer(object):
    def __init__(self,agent,environment,number_of_experts,
                 automaton_encoding_size, tg_reward, num_colors = 1, goal_reward_reduction_rate = 0.65):

        self.number_of_experts = number_of_experts
        self.automaton_encoding_size = automaton_encoding_size
        self.agent = agent
        self.environment = environment
        self.num_colors = num_colors
        self.goal_reward_reduction_rate = goal_reward_reduction_rate # how much of the first goals reward is given to the final one
        self.reward_step = np.round(float(tg_reward)*(1/self.num_colors)*goal_reward_reduction_rate, 2)
        # self.using_discount = True if (self.agent.spec['discount'] > 0 and self.agent.spec['discount'] < 1) else False

    def get_reward_from_automaton_state(self, reward, current_automaton_state, previous_automaton_state, terminal):
        for i in range(1, self.num_colors+1):
            if current_automaton_state == i and previous_automaton_state == i-1:
                reward += self.reward_step
                if current_automaton_state == self.num_colors:
                    terminal = True
                    reward += self.reward_step*(1-self.goal_reward_reduction_rate)*(self.num_colors)
                return reward, terminal
        return reward, terminal

    def train(self,episodes = 1000):
        num_time_visited_goal = 0
        cum_reward = 0.0
        agent = self.agent
        environment = self.environment
        reward_trend = 0.0
        old_reward_trend = 0.0
        pbar = tqdm(range(episodes),desc='[Training]',leave = True)
        try:
            for episode in pbar:
                if episode%1000 == 0: print("test")
                terminal = False

                #I obtain the obs and the automaton state to begin with
                states = environment.reset()

                #automaton_state = get_automaton_state_from_encoding(states['gymtpl1'], self.number_of_experts, self.automaton_encoding_size)
                automaton_state = environment._environment.environment.get_automaton_state()

                #I set the initial parameters to launch the training
                prevAutState = 0
                #Save the reward that you reach in the episode inside a linked list. This will be used for nice plots in the report.
                ep_reward = 0.0
                #agent internals
                internals = agent.initial_internals()

                while not terminal:
                    #I start the training setting the actions
                    actions = agent.act(states=states)

                    exploration = agent.model.exploration.value().numpy()
                    if 'ppo' in self.agent.__module__:
                        lr = None
                    else:
                        lr = agent.model.optimizer.learning_rate.value().numpy()

                    #I execute(?) the environment obtaining the states, the reward and if Im in a terminal condition or not
                    states, terminal, reward = environment.execute(actions=actions)
                    
                    #Extract gym sapientino state and the state of the automaton.
                    #automaton_state = get_automaton_state_from_encoding(states['gymtpl1'], self.number_of_experts, self.automaton_encoding_size)
                    
                    automaton_state = environment._environment.environment.get_automaton_state()
                    
                    """
                        Reward shaping.
                    """
                    # if not self.using_discount and terminal == 2:
                    #     reward = -self.reward_step
                    reward, terminal = self.get_reward_from_automaton_state(reward, automaton_state, prevAutState, terminal)
                    

                    #I update the previous state with the state in which I was in this training cycle,regardless of the fact
                    #that I have transitated in a new relevant state.
                    prevAutState = automaton_state


                    #Update the cumulative reward during the training.
                    cum_reward += reward

                    #Update the episode reward during the training
                    ep_reward += reward
                    
                    if episode%100==0: 
                        #old_reward_trend = reward_trend
                        reward_trend = 0.0
                    
                    if terminal == True: 
                        num_time_visited_goal += 1
                        reward_trend += 1
                 
                    pbar.set_postfix({#'reward': reward, 
                                      #'ep_reward': ep_reward, 
                                      'total_reward': cum_reward,
                                      'lr': lr,
                                      'expl':exploration, 
                                      'visited_goal_for_n_time': num_time_visited_goal,
                                      #'goal in 100ep': reward_trend
                                      })

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

