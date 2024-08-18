import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

class DQNAgent(base_agent.BaseAgent):
    def __init__(self, state_size=10, action_size=11):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.action_map = {
            0: self.select_command,
            1: self.train_scv,
            2: self.select_scv,
            3: self.move_scv,
            4: self.build_baracks,
            5: self.select_barracks,
            6: self.build_supply,
            7: self.train_marine,
            8: self.select_all_marines,
            9: self.move_marines,
            10: self.no_op
        }

    def _build_model(self):
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def step(self, obs):
        super(DQNAgent, self).step(obs)
        state = self.get_state(obs)
        action = self.act(state)
        return self.action_map[action](obs)

    def get_state(self, obs):
        minerals = obs.observation.player.minerals
        food_used = obs.observation.player.food_used
        food_cap = obs.observation.player.food_cap
        supply_remaining = food_cap - food_used
        scvs_count = len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.SCV])
        marines_count = len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.Marine])
        barracks_count = len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.Barracks and unit.build_progress == 100])
        barracks_building = len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.Barracks and unit.build_progress < 100])
        sd_count = len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.SupplyDepot and unit.build_progress == 100])
        sd_building = len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.SupplyDepot and unit.build_progress < 100])
        state = np.array([
            minerals,
            food_used,
            food_cap,
            supply_remaining,
            scvs_count,
            marines_count,
            barracks_count,
            barracks_building,
            sd_count,
            sd_building
        ], dtype=np.float32)
        return state.reshape(1, -1)  # Reshape to (1, state_size) for Keras model input

    # Implement your action methods here (select_command, train_scv, etc.)
    def select_command(self, obs):
        commands = self.get_units_by_type(obs, units.Terran.CommandCenter)
        if len(commands) > 0:
            command = random.choice(commands)
            return actions.FUNCTIONS.select_point("select", (command.x,command.y))
        return actions.FUNCTIONS.no_op()

    def train_scv(self, obs):
        if self.unit_type_is_selected(obs, units.Terran.CommandCenter):
            if (actions.FUNCTIONS.Train_SCV_quick.id in
                        obs.observation.available_actions):
                return actions.FUNCTIONS.Train_SCV_quick('now')
        else:
            return self.select_command(obs)
        return actions.FUNCTIONS.no_op()

    def select_scv(self, obs):
        scvs = self.get_units_by_type(obs, units.Terran.SCV)
        if len(scvs) > 0:
            scv = random.choice(scvs)
            if scv.x < 0:
                scv.x = 0
            if scv.y < 0:
                scv.y = 0
            if scv.x > 83:
                scv.x = 83
            if scv.y > 83:
                scv.y = 83
            return actions.FUNCTIONS.select_point("select", (scv.x,scv.y))
            
        return actions.FUNCTIONS.no_op()

    def move_scv(self, obs):
        position = (random.randint(0,83), random.randint(0,83))
        if self.unit_type_is_selected(obs, units.Terran.SCV) and (actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions):
            return actions.FUNCTIONS.Move_screen("now", position)
        else:
            self.select_scv(obs)
        return actions.FUNCTIONS.no_op()

    def build_baracks(self, obs):
        position = (random.randint(0,83), random.randint(0,83))
        if self.unit_type_is_selected(obs, units.Terran.SCV):
            if (actions.FUNCTIONS.Build_Barracks_screen.id in
                    obs.observation.available_actions):
                return actions.FUNCTIONS.Build_Barracks_screen('now', position)
        else:
            return self.select_scv(obs)
            
        return actions.FUNCTIONS.no_op()

    def select_barracks(self, obs):
        barracks = self.get_units_by_type(obs, units.Terran.Barracks)
        if len(barracks) > 0:
            barrack = random.choice(barracks)
            if barrack.x < 0:
                barrack.x = 0
            if barrack.y < 0:
                barrack.y = 0
            if barrack.x > 83:
                barrack.x = 83
            if barrack.y > 83:
                barrack.y = 83
            return actions.FUNCTIONS.select_point("select", (barrack.x, barrack.y))
        
        return actions.FUNCTIONS.no_op()

    def build_supply(self, obs):
        position = (random.randint(0,83), random.randint(0,83))
        if self.unit_type_is_selected(obs, units.Terran.SCV):
            if (actions.FUNCTIONS.Build_SupplyDepot_screen.id in
                    obs.observation.available_actions):
                return actions.FUNCTIONS.Build_SupplyDepot_screen('now', position)
        else:
            return self.select_scv(obs)
        return actions.FUNCTIONS.no_op()

    def train_marine(self, obs):
        if self.unit_type_is_selected(obs, units.Terran.Barracks):
             if (actions.FUNCTIONS.Train_Marine_quick.id in
                    obs.observation.available_actions):
                return actions.FUNCTIONS.Train_Marine_quick('now')
        else:
            self.select_barracks(obs)
        return actions.FUNCTIONS.no_op()

    def select_all_marines(self, obs):
        marines = self.get_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 0:
            marine = random.choice(marines)
            if marine.x < 0:
                marine.x = 0
            if marine.y < 0:
                marine.y = 0
            if marine.x > 83:
                marine.x = 83
            if marine.y > 83:
                marine.y = 83
            return actions.FUNCTIONS.select_point('select_all_type', (marine.x, marine.y))
        return actions.FUNCTIONS.no_op()

    def move_marines(self, obs):
        position = (random.randint(0,83), random.randint(0,83))
        if self.unit_type_is_selected(obs, units.Terran.Marine) and (actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions):
            return actions.FUNCTIONS.Move_screen("now", position)
        else:
            return self.select_all_marines(obs)
        return actions.FUNCTIONS.no_op()
    
    def no_op(self, obs):
        return actions.FUNCTIONS.no_op()

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