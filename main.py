import random
import time
import argparse
import configparser
import os
from gym_sapientino_case.env import SapientinoCase
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)
from gym.wrappers import TimeLimit
from utils import get_config

def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description='Train an agent on SapientinoCase.')
    parser.add_argument('--config_file', default='config1.cfg')
    args = parser.parse_args()
    config_file = args.config_file

    configuration = configparser.ConfigParser()
    configuration.read(os.path.join('./configs/',config_file))

    environment_config = get_config(configuration)

    env = SapientinoCase( 
        conf = environment_config, 
        reward_ldlf = None, 
        logdir = './experiments'
        )

        # Limit the length of the episode
    env = TimeLimit(env, 100)

    # interactive = False

    # Episodes
    while True:

        # Init episode
        obs = env.reset()
        done = False
        cum_reward = 0.0

        # Print
        print(f"\n> Env reset.\nInitial observation {obs}")

        while not done:
            # Render
            env.render()

            # Compute action
            # if interactive:
            #     try:
            #         action = int(input("Next action: "))
            #         if action < 0:
            #             print("Reset")
            #             env.reset()
            #             continue
            #         if action >= env.action_space.n:
            #             continue
            #     except ValueError:
            #         continue
            # else:
            action = random.randint(0, env.action_space.n - 1)

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
            time.sleep(0.1)
    

if __name__ == '__main__':
    main()

    

