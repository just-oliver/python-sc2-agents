import random
import numpy as np
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

# Actions
NO_OP = actions.FUNCTIONS.no_op.id
SELECT_POINT = actions.FUNCTIONS.select_point.id
HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
HARVEST_RETURN = actions.FUNCTIONS.Harvest_Return_quick.id
TRAIN_PROBE = actions.FUNCTIONS.Train_Probe_quick.id
BUILD_ASSIMILATOR = actions.FUNCTIONS.Build_Assimilator_screen.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_PROBE = units.Protoss.Probe
_NEXUS = units.Protoss.Nexus
_MINERAL_FIELD = units.Neutral.MineralField
_VESPENE_GEYSER = units.Neutral.VespeneGeyser
_ASSIMILATOR = units.Protoss.Assimilator

class CollectMineralsAndGasAgent(base_agent.BaseAgent):
    def __init__(self):
        super(CollectMineralsAndGasAgent, self).__init__()
        self.base_top_left = None

    def step(self, obs):
        super(CollectMineralsAndGasAgent, self).step(obs)

        # Check if game has just started to locate base position
        if obs.first():
            nexus = self.get_units_by_type(obs, _NEXUS)[0]
            self.base_top_left = (nexus.x < 32)

        # Get available probes
        probes = self.get_units_by_type(obs, _PROBE)
        if len(probes) == 0:
            return actions.FunctionCall(NO_OP, [])

        probe = random.choice(probes)

        # Select the probe if not already selected
        if obs.observation.single_select is None or obs.observation.single_select[0].unit_type != _PROBE:
            return actions.FunctionCall(SELECT_POINT, [actions.NOT_QUEUED, (probe.x, probe.y)])

        # Check for mineral fields
        mineral_fields = self.get_units_by_type(obs, _MINERAL_FIELD)
        if len(mineral_fields) > 0:
            mf = random.choice(mineral_fields)
            return actions.FunctionCall(HARVEST_GATHER, [actions.NOT_QUEUED, (mf.x, mf.y)])

        # Check for assimilators
        assimilators = self.get_units_by_type(obs, _ASSIMILATOR)
        if len(assimilators) > 0 and probe.orders == []:
            assimilator = random.choice(assimilators)
            return actions.FunctionCall(HARVEST_GATHER, [actions.NOT_QUEUED, (assimilator.x, assimilator.y)])

        return actions.FunctionCall(NO_OP, [])

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.raw_units if unit.unit_type == unit_type]

    def get_mineral_fields(self, obs):
        return self.get_units_by_type(obs, _MINERAL_FIELD)

    def get_assimilators(self, obs):
        return self.get_units_by_type(obs, _ASSIMILATOR)