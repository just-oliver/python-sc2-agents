from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from actor_critic import ActorCriticNetwork


class Agent(base_agent.BaseAgent):
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=10):
        super(Agent, self).__init__()
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.action_space = [i for i in range(self.n_actions)]
        #self.actor_critic = ActorCriticNetwork()

    def choose_action(self, obs):
        state = tf.convert_to_tensor
    def step(self, obs):
        super(Agent, self).step(obs)
        if self.steps == 1:
            print(obs.observation['feature_screen'][features.SCREEN_FEATURES.player_relative.index][0])
        if self.steps == 16:
            return self.select_scv(obs)
        if obs.observation.player.minerals >= 100 and len(self.get_units_by_type(obs,  units.Terran.SupplyDepot)) == 0:
            return self.build_supply(obs, (40,60))
        if obs.observation.player.minerals >= 150 and len(self.get_units_by_type(obs, units.Terran.SupplyDepot)) > 0 and len(self.get_units_by_type(obs,  units.Terran.Barracks)) == 0:
            return self.build_baracks(obs, (40, 20))
        if len(self.get_units_by_type(obs, units.Terran.Barracks)) > 0 and (not self.unit_type_is_selected(obs, units.Terran.Barracks)):
            return self.select_barracks(obs)

        if self.unit_type_is_selected(obs, units.Terran.Barracks):
            return self.train_marine(obs)
        return actions.FUNCTIONS.no_op()

    def select_command(self, obs, i=0):
        commands = self.get_units_by_type(obs, units.Terran.CommandCenter)
        if len(commands) > 0:
            command = commands[i]
            return actions.FUNCTIONS.select_point("select", (command.x,command.y))
        return actions.FUNCTIONS.no_op()

    def train_scv(self, obs):
        pass
    def select_scv(self, obs, i=0):
        scvs = self.get_units_by_type(obs, units.Terran.SCV)
        if len(scvs) > 0:
            scv = scvs[i]
            return actions.FUNCTIONS.select_point("select", (scv.x,scv.y))
        return actions.FUNCTIONS.no_op()

    def build_baracks(self, obs, position):
        if self.unit_type_is_selected(obs, units.Terran.SCV):
            if (actions.FUNCTIONS.Build_Barracks_screen.id in
                    obs.observation.available_actions):
                return actions.FUNCTIONS.Build_Barracks_screen('now', position)
        return actions.FUNCTIONS.no_op()

    def select_barracks(self, obs, i=0):
        barracks = self.get_units_by_type(obs, units.Terran.Barracks)
        if len(barracks) > 0:
            barrack = barracks[i]
            return actions.FUNCTIONS.select_point("select", (barrack.x, barrack.y))
        return actions.FUNCTIONS.no_op()

    def build_supply(self, obs, position):
        if self.unit_type_is_selected(obs, units.Terran.SCV):
            if (actions.FUNCTIONS.Build_SupplyDepot_screen.id in
                    obs.observation.available_actions):
                return actions.FUNCTIONS.Build_SupplyDepot_screen('now', position)
        return actions.FUNCTIONS.no_op()

    def train_marine(self, obs):
        if self.unit_type_is_selected(obs, units.Terran.Barracks):
             if (actions.FUNCTIONS.Train_Marine_quick.id in
                    obs.observation.available_actions):
                return actions.FUNCTIONS.Train_Marine_quick('now')
        return actions.FUNCTIONS.no_op()

    def select_all_marines():
        return actions.FUNCTIONS.no_op()

    def move_marines():
        pass

    def policy_model():
        pass
    def value_model():
        pass

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
            obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
            obs.observation.multi_select[0].unit_type == unit_type):
            return True
        return False

