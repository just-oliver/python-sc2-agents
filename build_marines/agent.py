class Agent:
    class PyAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(PyAgent, self).step(obs)
        return actions.FUNCTIONS.no_op()
