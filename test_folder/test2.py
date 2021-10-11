import random
import time
import argparse
from gym.wrappers import TimeLimit
from gym_sapientino_case.env import SapientinoCase
from tensorforce import agents, environments
from tensorforce.agents.agent import Agent
from tensorforce.environments.environment import Environment

def main():
    

    env = TimeLimit(SapientinoCase(logdir="."), 30)

    obs = env.reset()
    done = False
    cum_reward = 0.0

    # Print
    print(f"\n> Env reset.\nInitial observation {obs}")

    while not done:
        # Render
        env.render()

        action = random.randint(0, env.action_space.n - 1) #env.action_space.n = 6

        # Move env
        obs, reward, done, _ = env.step(action)
        cum_reward += reward

        # Print
        print(
            "Step.",
            f"Action {action}",
            f"Observation {obs}",
            f"Reward {reward}",
            f"Done {done}",
            sep="\n  ",
        )

        # Let us see the screen
        #time.sleep(0.1)



def test():

    env = TimeLimit(SapientinoCase(logdir="."), 30)
    
    agent = Agent.create(agent='double_dqn', environment=env ,batch_size=64, memory= 600)


    states = env.reset()

    actions = agent.act(states)

    states, terminal, reward = Environment.step(actions=actions)
    num_updates = agent.observe(terminal=terminal, reward=reward)
    
    #train(environment,states,agent)
    #test(environment,states,agent)

    # Close agent and environment
    agent.close()
    env.close()


if __name__ == "__main__":
    test()
