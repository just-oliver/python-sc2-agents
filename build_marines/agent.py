#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Oliver Just-De Bleser
"""


from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from actor_critic import ActorCriticNetwork
import numpy as np
import tensorflow_probability as tfp


class Agent(base_agent.BaseAgent):
    def __init__(self, alpha=1e-4, gamma=0.99, n_actions=11, train=True):
        super(Agent, self).__init__()
        self.train = train
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.prev_action = None
        self.action_space = [i for i in range(self.n_actions)]
        self.actor_critic = ActorCriticNetwork(n_actions=n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))
        self.prev_scalar_state = None
        self.prev_spacial_state = None
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
        self.punishment = 0

    def choose_action(self, scalar_state, spacial_state):
        value, probs_action, x, y = self.actor_critic(scalar_state, spacial_state)
        action_probabilities = tfp.distributions.Categorical(probs=probs_action)
        action = tf.clip_by_value(action_probabilities.sample(), 0, 10)
        self.action = action
        self.x = x
        self.y = y
        try:
            x_pos = int(x * 83)
            y_pos = int(y * 83)
        except:
            print(f'nan error x:{x}, y:{y}', x, y)
            x_pos = random.randint(0,83)
            y_pos = random.randint(0,83)
        return action.numpy()[0], x_pos, y_pos
    
    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)
    
    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)
    
    def learn(self, scalar_state, spacial_state, reward, done):
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            state_value, action_probs, x, y = self.actor_critic(scalar_state, spacial_state)
            state_value_, _, _, _ = self.actor_critic(self.prev_scalar_state, self.prev_spacial_state)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)
            
            action_dist = tfp.distributions.Categorical(probs=action_probs)
            log_prob_action = action_dist.log_prob(self.action)
            x_loss = tf.square(x - self.x)
            y_loss = tf.square(y - self.y)
            delta = reward + self.gamma * state_value_ * (1 - int(done)) - state_value
            actor_loss = -log_prob_action * delta + 0.5 * (x_loss + y_loss)
            critic_loss = delta**2
            total_loss = actor_loss + critic_loss
        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(gradient, self.actor_critic.trainable_variables))
    
    def step(self, obs):
        super(Agent, self).step(obs)
        if obs.reward == 0:
            self.punishment = 0.1
        # creating our state space to feed to the a Neural network
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

        scalar_observations = np.array([
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
        ])
        # Assuming obs is your current observation
        feature_screen = obs.observation['feature_screen']
        unit_type = np.array(feature_screen[features.SCREEN_FEATURES.unit_type.index])
        command_screen = (unit_type == units.Terran.CommandCenter).astype(int)
        marine_screen = (unit_type == units.Terran.Marine).astype(int)
        scv_screen = (unit_type == units.Terran.SCV).astype(int)
        mineral_screen = (unit_type == units.Neutral.MineralField).astype(int)
        barrack_screen = (unit_type == units.Terran.Barracks).astype(int)
        selected = np.array(feature_screen[features.SCREEN_FEATURES.selected.index])
        unit_density = np.array(feature_screen[features.SCREEN_FEATURES.unit_density.index])
        active = np.array(feature_screen[features.SCREEN_FEATURES.active.index])
        pathable = np.array(feature_screen[features.SCREEN_FEATURES.pathable.index])
        
        spacial_observations = np.stack([
            unit_type,
            command_screen,
            marine_screen,
            scv_screen,
            mineral_screen,
            barrack_screen,
            selected,
            unit_density,
            active,
            pathable
        ], axis=-1)
        
        scalar_state = tf.convert_to_tensor([scalar_observations])
        spacial_state = tf.convert_to_tensor([spacial_observations], dtype=tf.float16)
        
        action, x, y = self.choose_action(scalar_state, spacial_state)
        if self.prev_action and self.train:
            self.learn(scalar_state, spacial_state, obs.reward - self.punishment, obs.last())
        self.prev_action = action
        self.prev_scalar_state = scalar_state
        self.prev_spacial_state = spacial_state
        self.punishment = 0
        if self.steps % 16 == 0:
            print(self.action_map[action], x, y)
        if action in [3, 4, 6, 9]:
            return self.action_map[action](obs, (x,y))
        return self.action_map[action](obs)

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
            return actions.FUNCTIONS.select_point("select", (scv.x,scv.y))
        return actions.FUNCTIONS.no_op()

    def move_scv(self, obs, position):
        if self.unit_type_is_selected(obs, units.Terran.SCV):
            return actions.FUNCTIONS.Move_screen("now", position)
        else:
            self.select_scv(obs)
        return actions.FUNCTIONS.no_op()

    def build_baracks(self, obs, position):
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
            return actions.FUNCTIONS.select_point("select", (barrack.x, barrack.y))
        
        return actions.FUNCTIONS.no_op()

    def build_supply(self, obs, position):
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
            return actions.FUNCTIONS.select_point('select_all_type', (marine.x, marine.y))
        return actions.FUNCTIONS.no_op()

    def move_marines(self, obs, position):
        if self.unit_type_is_selected(obs, units.Terran.Marine):
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
    
