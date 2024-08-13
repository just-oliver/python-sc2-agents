import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features


class CollectMineralShardsAgent(base_agent.BaseAgent):
  """Deep RL agent created to play/solve the CollectMineralShards map."""

  def setup(self, obs_spec, action_spec):
    super(CollectMineralShardsAgent, self).setup(obs_spec, action_spec)
    # TODO override any setup - done once on env initialization

  def reset(self):
    super(CollectMineralShardsAgent, self).reset()
    # TODO override any agent reset at the beginning of an episode

  def step(self, obs):
    super(CollectMineralShards, self).step(obs)
    # TODO implement the deep RL here