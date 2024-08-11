import random
import gym
from gym import spaces
import numpy as np
from pysc2.lib import actions

from src.collect_minerals_and_gas.agent import _MINERAL_FIELD, _PLAYER_RELATIVE, SELECT_POINT, NO_OP, HARVEST_GATHER, _ASSIMILATOR, _PROBE
from pysc2.env import sc2_env

class SC2Env(gym.Env):
    def __init__(self):
        super(SC2Env, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(4)  # Example action space size
        self.observation_space = spaces.Box(low=0, high=255, shape=(64, 64, 1), dtype=np.uint8)

        # Initialize SC2 environment
        self.env = sc2_env.SC2Env(
            map_name="CollectMineralsAndGas",
            players=[sc2_env.Agent(sc2_env.Race.protoss)],
            agent_interface_format=sc2_env.parse_agent_interface_format(
                feature_screen=64,
                feature_minimap=64,
                use_feature_units=True
            ),
            step_mul=8,
            game_steps_per_episode=0,
            visualize=False
        )

        self.current_obs = None

    def reset(self):
        self.current_obs = self.env.reset()
        return self._get_observation()

    def step(self, action):
        agent_action = self._convert_action(action)
        self.current_obs = self.env.step([agent_action])
        obs = self._get_observation()
        reward = self._get_reward()
        done = self._get_done()
        info = {}

        return obs, reward, done, info

    def _get_observation(self):
        # Extract screen or other features
        return self.current_obs[0].observation["feature_screen"][_PLAYER_RELATIVE]

    def _get_reward(self):
        return self.current_obs[0].reward

    def _get_done(self):
        return self.current_obs[0].last()

    def _convert_action(self, action):
        # Map the RL action to a SC2 action
        # For simplicity, assume action space size is 4: NO_OP, select probe, harvest mineral, harvest gas
        if action == 0:
            return actions.FunctionCall(NO_OP, [])
        elif action == 1:
            # Select a probe
            probes = self._get_units_by_type(_PROBE)
            if probes:
                probe = random.choice(probes)
                return actions.FunctionCall(SELECT_POINT, [0, (probe.x, probe.y)])
        elif action == 2:
            # Harvest mineral
            mineral_fields = self._get_units_by_type(_MINERAL_FIELD)
            if mineral_fields:
                mf = random.choice(mineral_fields)
                return actions.FunctionCall(HARVEST_GATHER, [0, (mf.x, mf.y)])
        elif action == 3:
            # Harvest gas
            assimilators = self._get_units_by_type(_ASSIMILATOR)
            if assimilators:
                assimilator = random.choice(assimilators)
                return actions.FunctionCall(HARVEST_GATHER, [0, (assimilator.x, assimilator.y)])

        return actions.FunctionCall(NO_OP, [])

    def _get_units_by_type(self, unit_type):
        return [unit for unit in self.current_obs[0].observation.raw_units if unit.unit_type == unit_type]

    def render(self, mode='human'):
        if mode == 'human':
            # Rendering for human mode is typically done via the visualization flag
            self.env.render()
        else:
            super(SC2Env, self).render(mode=mode)

    def close(self):
        self.env.close()