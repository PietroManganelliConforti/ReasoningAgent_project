import argparse
import configparser
import os
from gym_sapientino_case.env import SapientinoCase
from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)

def main():

    # Argument parsing
    parser = argparse.ArgumentParser(description='Train an agent on SapientinoCase.')
    parser.add_argument('--config_file', default='config1.cfg')
    args = parser.parse_args()
    config_file = args.config_file

    configuration = configparser.ConfigParser()
    configuration.read(os.path.join('./configs/',config_file))
    env_cfg = configuration['ENVIRONMENT']
    colors = env_cfg['colors'].replace(' ', '').split(',')
    # agent_conf = SapientinoAgentConfiguration(
    #             initial_position=,
    #             commands=,
    #             angular_speed=,
    #             acceleration=,
    #             max_velocity=,
    #             min_velocity=,
    #         )
    # conf = SapientinoConfiguration(
    #     agent_configs=(agent_conf,),
    #     grid_map=,
    #     reward_outside_grid=,
    #     reward_duplicate_beep=,
    #     reward_per_step=,
    # )
    # env = SapientinoCase( conf, reward_ldlf, logdir)
    env = SapientinoCase()
    

if __name__ == '__main__':
    main()

    

