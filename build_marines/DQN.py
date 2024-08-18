import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Concatenate, Flatten, Dropout
from collections import deque
from tensorflow.keras.optimizers import Adam
import random
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
import numpy as np
import os
import tensorflow as tf

def min_max_scale(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

class DQN(keras.Model):
    def __init__(self, n_actions, fc1_dims=64, fc2_dims=64,
                 name='DQN', chkpt_dir='tmp/DQN'):
        super(DQN, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ac.weights.h5')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.n_actions = n_actions
        self.model_name = name
        self.dense1 = Dense(fc1_dims, activation='relu')
        self.dense2 = Dense(fc2_dims, activation='relu')
        self.output_layer = Dense(n_actions, activation='linear')
        
    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

class DQNAgent(base_agent.BaseAgent):
    def __init__(self, state_size=9, n_actions=6, learning_rate=1e-3, load_models=False, train=True, model_names="DQN", epsilon_decay=0.995):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.n_actions = n_actions
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        if load_models:
            self.load_models(model_names)
        else:
            self._build_models(model_names)
        self.frame_skip = 8 # predict every 8th step - twice per sec
        self.update_frequency = 1000
        self.prev_state = None
        self.prev_action = None
        self.action_map = {
            #0: self.select_command,
            #1: self.train_scv,
            0: self.select_scv,
            #3: self.move_scv,
            1: self.build_baracks,
            2: self.select_barracks,
            3: self.build_supply,
            4: self.train_marine,
            #8: self.select_all_marines,
            #9: self.move_marines,
            5: self.no_op
        }
        self.steps_to_sync = 10
        
    
    def _build_models(self, model_names):
        self.policy_model = DQN(self.n_actions, name=model_names)
        self.policy_model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        self.target_model = DQN(self.n_actions, name=model_names)
        self.target_model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        self.target_model.set_weights(self.policy_model.get_weights())
    
    def step(self, obs):
        super(DQNAgent, self).step(obs)
        #if self.steps % 8 == 0:
        minerals = min_max_scale(obs.observation.player.minerals, 0, 5000)
        supply_remaining = min_max_scale(obs.observation.player.food_cap-obs.observation.player.food_used, 0, 20)
        scvs_count = min_max_scale(len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.SCV]), 0, 16)
        marines_count = min_max_scale(len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.Marine]), 0, 100)
        barracks_count = min_max_scale(len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.Barracks and unit.build_progress == 100]),0, 8)
        barracks_building = min_max_scale(len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.Barracks and unit.build_progress < 100]), 0, 8)
        sd_count = min_max_scale(len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.SupplyDepot and unit.build_progress == 100]), 0, 10)
        sd_building = min_max_scale(len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.SupplyDepot and unit.build_progress < 100]), 0, 10)
        barracks_selected = int(self.unit_type_is_selected(obs, units.Terran.Barracks))
        state = np.array([
            minerals,
            supply_remaining,
            scvs_count,
            marines_count,
            barracks_count,
            barracks_building,
            sd_count,
            sd_building,
            barracks_selected
        ])
        
        if self.prev_state is not None:
            self.remember(obs.reward, state, obs.last())
        return self.act(state, obs)
        # else:
        #     return actions.FUNCTIONS.no_op()
        
    def act(self, state, obs):
        self.prev_state = state
        if np.random.rand() <= self.epsilon:
            action_index = random.randint(0, self.n_actions - 1)
        else:
            state_tensor = tf.convert_to_tensor(state[np.newaxis, ...], dtype=tf.float32)
            act_values = self.policy_model(state_tensor)
            action_index = np.argmax(act_values[0])
        
        self.prev_action = action_index
        action_function = self.action_map[action_index]
        return action_function(obs)
    
    def remember(self, reward, state, done):
        self.memory.append((self.prev_state, self.prev_action, reward,
                            state, done))
    
    def replay(self, batch_size):
        mini_batch = random.sample(self.memory, batch_size)
        states = np.array([transition[0] for transition in mini_batch])
        actions = np.array([transition[1] for transition in mini_batch])
        rewards = np.array([transition[2] for transition in mini_batch])
        next_states = np.array([transition[3] for transition in mini_batch])
        dones = np.array([transition[4] for transition in mini_batch])

        targets = rewards + self.gamma * np.max(self.target_model.predict(next_states), axis=1) * (1 - dones)
        target_f = self.policy_model.predict(states)
        for i, action in enumerate(actions):
            target_f[i][action] = targets[i]

        self.policy_model.fit(states, target_f, epochs=1, verbose=0, batch_size=batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
            
    
    # def select_command(self, obs):
    #     commands = self.get_units_by_type(obs, units.Terran.CommandCenter)
    #     if len(commands) > 0:
    #         command = random.choice(commands)
    #         if command.x < 0:
    #             command.x = 0
    #         if command.y < 0:
    #             command.y = 0
    #         if command.x > 83:
    #             command.x = 83
    #         if command.y > 83:
    #             command.y = 83
    #         return actions.FUNCTIONS.select_point("select", (command.x,command.y))
    #     return actions.FUNCTIONS.no_op()

    # def train_scv(self, obs):
    #     if self.unit_type_is_selected(obs, units.Terran.CommandCenter):
    #         if (actions.FUNCTIONS.Train_SCV_quick.id in
    #                     obs.observation.available_actions):
    #             return actions.FUNCTIONS.Train_SCV_quick('now')
    #     else:
    #         return self.select_command(obs)
    #     return actions.FUNCTIONS.no_op()

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

    # def move_scv(self, obs):
    #     position = (random.randint(0,83), random.randint(0,83))
    #     if self.unit_type_is_selected(obs, units.Terran.SCV) and (actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions):
    #         return actions.FUNCTIONS.Move_screen("now", position)
    #     else:
    #         self.select_scv(obs)
    #     return actions.FUNCTIONS.no_op()

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

    # def select_all_marines(self, obs):
    #     marines = self.get_units_by_type(obs, units.Terran.Marine)
    #     if len(marines) > 0:
    #         marine = random.choice(marines)
    #         if marine.x < 0:
    #             marine.x = 0
    #         if marine.y < 0:
    #             marine.y = 0
    #         if marine.x > 83:
    #             marine.x = 83
    #         if marine.y > 83:
    #             marine.y = 83
    #         return actions.FUNCTIONS.select_point('select_all_type', (marine.x, marine.y))
    #     return actions.FUNCTIONS.no_op()

    # def move_marines(self, obs):
    #     position = (random.randint(0,83), random.randint(0,83))
    #     if self.unit_type_is_selected(obs, units.Terran.Marine) and (actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions):
    #         return actions.FUNCTIONS.Move_screen("now", position)
    #     else:
    #         return self.select_all_marines(obs)
    #     return actions.FUNCTIONS.no_op()
    
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
    
    def save_models(self):
        print('... saving models ...')
        self.policy_model.save_weights(self.policy_model.checkpoint_file)
    
    def load_models(self, model_names):
        print('... loading models ...')
        self.policy_model = DQN(self.n_actions, name=model_names)
        self.policy_model.load_weights(self.policy_model.checkpoint_file)
        self.policy_model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        self.target_model = DQN(self.n_actions, name=model_names)
        self.target_model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        self.target_model.set_weights(self.policy_model.get_weights())
    