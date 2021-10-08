from tensorforce.agents import Agent
from tensorforce.environments import Environment


from one_hot import one_hot_encode
from tqdm.auto import tqdm
import numpy as np



DEBUG = False

def get_automaton_state_from_encoding(encoding, num_expert, encoding_size):
    if np.max(encoding) == 0: return num_expert
    automaton_state = np.argmax(encoding)/(encoding_size/num_expert)
    return int(automaton_state)


class Trainer(object):
    def __init__(self,agent,environment,number_of_experts,
                 automaton_encoding_size, num_colors = 2,):

        """
        Desc: class that implements the non markovian training (multiple colors for gym sapientino).
        Keep in mind that the class instantiates the agent according to the parameter dictionary stored inside
        "agent_params" variable. The agent, and in particular the neural network, should be already non markovian.
        Args:
            @param agent: (tensorforce.agents.Agent) tensorforce agent (algrithm) that will be used to train the policy network (example: ppo, ddqn,dqn).
            @param environment: (tensorforce.environments.Environment) istance of the tensorforce/openAI gym environment used for training.
            @param num_state_automaton: (int) number of states of the goal state DFA.
            @automaton_state_encoding_size: (int) size of the binary encoding of the automaton state. See the report in report/pdf in section "Non markovian agent" for further details.
        """



        self.number_of_experts = number_of_experts
        self.automaton_encoding_size = automaton_encoding_size


        #Create both the agent and the environment that will be used a training time.
        self.agent = agent
        self.environment = environment


        self.num_colors = num_colors



        if DEBUG:
            print("\n################### Agent architecture ###################\n")
            architecture = self.agent.get_architecture()
            print(architecture)





    def train(self,episodes = 1000):
        cum_reward = 0.0
        agent = self.agent
        environment = self.environment
        pbar = tqdm(range(episodes),desc='training',leave = True)
        try:
            for episode in pbar:
                terminal = False

                #I obtain the obs and the automaton state to begin with
                states = environment.reset()

                automaton_state = get_automaton_state_from_encoding(states['gymtpl1'], self.number_of_experts, self.automaton_encoding_size)

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
                    automaton_state = get_automaton_state_from_encoding(states['gymtpl1'], self.number_of_experts, self.automaton_encoding_size)


                    """
                        Reward shaping.
                    """
                    #adapting the NonMarkovian class for a General markovian-or-not case with up to 4 states, written by matteo emanuele,
                    #with the new environment state modelled(no sink state).

                    #the second condition in the final, nested elif is not really needed, since when the state becomes the goal(aka the last state), 
                    # the terminal condition is reached.
                    if self.num_colors == 1:
                        if automaton_state == 1 and prevAutState==0:
                            reward = 500.0
                            print("Visited goal on episode: ", episode)
                            terminal = True
                            
                    elif self.num_colors == 2:

                        if automaton_state == 1 and prevAutState==0:
                            reward = 500.0

                        elif automaton_state == 2 and prevAutState==1:
                            reward = 500.0
                            print("Visited goal on episode: ", episode)
                            terminal = True

                    elif self.num_colors == 3:

                        if automaton_state == 1 and prevAutState==0:
                            reward = 500.0

                        elif automaton_state == 2 and prevAutState==1:
                            reward = 500.0

                        elif automaton_state ==3 and prevAutState == 2:
                            reward = 500.0
                            print("Visited goal on episode: ", episode)
                            terminal = True


                    elif self.num_colors == 4:

                        if automaton_state == 1 and prevAutState==0:
                            reward = 500.0

                        elif automaton_state == 2 and prevAutState==1:
                            reward = 500.0

                        elif automaton_state ==3 and prevAutState == 2:
                            reward = 500.0

                        elif automaton_state == 4 and prevAutState==3:
                            reward = 500.0
                            print("Visited goal on episode: ", episode)
                            terminal = True


                    #I update the previous state with the state in which I was in this training cycle,regardless of the fact
                    #that I have transitated in a new relevant state.
                    prevAutState = automaton_state


                    #Update the cumulative reward during the training.
                    cum_reward += reward

                    #Update the episode reward during the training
                    ep_reward += reward
                    pbar.set_postfix({'current_reward': reward, 'episode_reward': ep_reward, 'total_reward': cum_reward})

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

