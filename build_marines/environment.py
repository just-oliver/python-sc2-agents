#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Oliver Just-De Bleser
"""


from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
from agent import Agent
import tensorflow as tf

def main(unused_argv):
    agent = Agent()
    try:
        while True:
            with sc2_env.SC2Env(
                    map_name="BuildMarines",
                    players=[sc2_env.Agent(sc2_env.Race.terran)],
                    agent_interface_format=features.AgentInterfaceFormat(
                        feature_dimensions=features.Dimensions(screen=84, minimap=64),
                        use_feature_units=True),
                step_mul=16, # action 2x per second, set to 16 for action per second
                game_steps_per_episode=0,
                visualize=True) as env:
                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app.run(main)
