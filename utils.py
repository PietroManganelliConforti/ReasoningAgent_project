from gym_sapientino.core.configurations import (
    SapientinoAgentConfiguration,
    SapientinoConfiguration,
)
from gym_sapientino.core.actions import ContinuousCommand
import os
import re

def get_agent_configuration(config):
    initial_position_x = float(config['initial_position_x'])
    initial_position_y = float(config['initial_position_y'])
    angular_speed = float(config['angular_acceleration'])
    acceleration = float(config['acceleration'])
    max_velocity = float(config['max_velocity'])
    min_velocity = float(config['min_velocity'])
    return SapientinoAgentConfiguration(
                initial_position= (initial_position_x, initial_position_y),
                commands=ContinuousCommand,
                angular_speed=angular_speed,
                acceleration=acceleration,
                max_velocity=max_velocity,
                min_velocity=min_velocity,
            )

def get_env_configuration(env_config, agent_config):
    grid_map_file = os.path.join('./maps/', env_config['map_file'])
    with open(grid_map_file) as f:
        grid_map = f.readlines()
        grid_map = ''.join(grid_map)
    reward_outside_grid = float(env_config['reward_outside_grid'])
    reward_duplicate_beep = float(env_config['reward_duplicate_beep'])
    reward_per_step = float(env_config['reward_per_step'])
    return SapientinoConfiguration(
        agent_configs=(agent_config,),
        grid_map=grid_map,
        reward_outside_grid=reward_outside_grid,
        reward_duplicate_beep=reward_duplicate_beep,
        reward_per_step=reward_per_step,
    )

def get_config(config):
    env_config = config['ENVIRONMENT']
    agent_config = get_agent_configuration(config['AGENT'])
    configuration = get_env_configuration(env_config, agent_config)
    return configuration

def get_colors(reward_ldlf):
    pattern_colors_on_ldlf = re.compile(r'![a-z]+*',re.IGNORECASE)
    colors = [ re.sub('!|*', '', x) for x in pattern_colors_on_ldlf.findall(reward_ldlf)]
    return colors