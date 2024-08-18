#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 14:53:44 2024

@author: deck
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Oliver Just-De Bleser
"""


from pysc2.env import sc2_env
from pysc2.lib import features
from absl import app
from DQN import DQNAgent
import numpy as np


def main(unused_argv):
    lr = 1e-4
    epsilon_decay = 0.95
    agent = DQNAgent(model_names=f'DQN_{lr}_{epsilon_decay}', learning_rate=lr, epsilon_decay=epsilon_decay)
    best_score = 0
    score_history = []
    load_checkpoint = False
    number_of_episodes = 500
    rewards_per_episode = np.zeros(number_of_episodes)
    batch_size = 32
    with open(f'DQN_data_{lr}_{epsilon_decay}.csv', 'w') as f:
        try:
            for i in range(number_of_episodes):
                with sc2_env.SC2Env(
                        map_name="BuildMarines",
                        players=[sc2_env.Agent(sc2_env.Race.terran)],
                        agent_interface_format=features.AgentInterfaceFormat(
                            feature_dimensions=features.Dimensions(screen=84, minimap=64),
                            use_feature_units=True),
                    step_mul=32, # action 2x per second, set to 16 for action per second
                    game_steps_per_episode=0,
                    visualize=False) as env:
                    agent.setup(env.observation_spec(), env.action_spec())
                    timesteps = env.reset()
                    agent.reset()
    
                    while True:
                        step_actions = [agent.step(timesteps[0])]
                        if len(agent.memory) > batch_size and agent.steps % agent.update_frequency == 0 and np.sum(rewards_per_episode) > 0:
                            agent.replay(batch_size)
                        if timesteps[0].last():
                            score_history.append(agent.reward)
                            avg_score = np.mean(score_history[-100:])
                            if avg_score > best_score:
                                best_score = avg_score
                                if not load_checkpoint:
                                    agent.save_models()
                            print('\n\n')
                            print('episode ', i, 'score %.1f' % agent.reward, 'avg_score %.1f' % avg_score)
                            print('\n\n')
                            f.write(f'{i}, {agent.reward}, {avg_score}\n')
                            rewards_per_episode[i] = agent.reward
                            break
                            
                        timesteps = env.step(step_actions)
                
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    app.run(main)
