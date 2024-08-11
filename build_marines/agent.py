from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

class Agent(base_agent.BaseAgent):
    def step(self, obs):
        super(Agent, self).step(obs)
        if self.steps % 120 == 0:
            print(obs.observation.available_actions)
            print(f"Action space: {self.action_spec}")
            print(f"Action space: {self.obs_spec}")
        return actions.FUNCTIONS.no_op()
