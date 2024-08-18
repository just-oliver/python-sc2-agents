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
from actor_critic import ActorCriticNetwork_nopos

import numpy as np
import tensorflow_probability as tfp


def min_max_scale(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

class Agent(base_agent.BaseAgent):
    def __init__(self, alpha=1e-5, gamma=0.99, n_actions=11, train=True):
        super(Agent, self).__init__()
        self.train = train
        self.gamma = gamma
        self.n_actions = n_actions
        self.action = None
        self.prev_action = None
        self.action_space = [i for i in range(self.n_actions)]
        self.actor_critic = ActorCriticNetwork_nopos(n_actions=n_actions)
        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))
        self.prev_scalar_state = None
        self.prev_spacial_state = None
        self.previous_units = None
        self.marine_training = False
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
        self.bonus = 0
    
    def is_marine_training(self, obs):
        if features.FeatureUnit.production_queue in obs.observation['feature_units']:
            # Iterate through all units
            for unit in obs.observation['feature_units']:
                # Check if the unit is a Barracks
                if unit.unit_type == units.Terran.Barracks:
                    # Check if the Barracks has a non-empty production queue
                    if unit.production_queue:
                        # Check if the unit in production is a Marine
                        if unit.production_queue[0] == units.Terran.Marine:
                            return True
        return False
    
    def choose_action(self, scalar_state, spacial_state):
        value, probs_action = self.actor_critic(scalar_state, spacial_state)
        # Ensure probs_action has the correct shape (batch_size, n_actions)
        probs_action = tf.reshape(probs_action, [-1, self.n_actions])
        action_probabilities = tfp.distributions.Categorical(probs=probs_action)
        action = action_probabilities.sample()
        # Clip the action after sampling to ensure it's within the valid range
        action = tf.clip_by_value(action, 0, self.n_actions - 1)
        self.action = action
        return action.numpy()[0]
    
    def save_models(self):
        print('... saving models ...')
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)
    
    def load_models(self):
        print('... loading models ...')
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)
    
    def learn(self, scalar_state, spacial_state, reward, done):
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            state_value, action_probs = self.actor_critic(scalar_state, spacial_state)
            state_value_, _ = self.actor_critic(self.prev_scalar_state, self.prev_spacial_state)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)
            
            action_dist = tfp.distributions.Categorical(probs=action_probs)
            log_prob_action = action_dist.log_prob(self.action)
            delta = reward + self.gamma * state_value_ * (1 - int(done)) - state_value
            actor_loss = -log_prob_action * delta 
            critic_loss = delta**2
            total_loss = actor_loss + critic_loss
        gradient = tape.gradient(total_loss, self.actor_critic.trainable_variables)
        gradient, _ = tf.clip_by_global_norm(gradient, 0.5)
        self.actor_critic.optimizer.apply_gradients(zip(gradient, self.actor_critic.trainable_variables))
    
    def step(self, obs):
        super(Agent, self).step(obs)
        if self.steps % 8 == 0:
            if self.marine_training:
                self.bonus = 1
                self.marine_training = False
            units_types = [unit.unit_type for unit in obs.observation['feature_units']]
            if units.Terran.SupplyDepot in units_types  and units.Terran.SupplyDepot not in self.prev_units:
                self.bonus = 1
            elif units.Terran.Barracks in units_types and units.Terran.Barracks not in self.prev_units:
                self.bonus = 1
            else:
                self.bonus = 0
            
            self.prev_units = units_types
            # if obs.reward == 0:
            #     self.punishment = 1e-5
            # creating our state space to feed to the a Neural network
            # scale input values between 0 and 1 to avoid exploding graidients
            minerals = min_max_scale(obs.observation.player.minerals, 0, 5000)
            supply_remaining = min_max_scale(obs.observation.player.food_cap-obs.observation.player.food_used, 0, 20)
            scvs_count = min_max_scale(len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.SCV]), 0, 16)
            marines_count = min_max_scale(len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.Marine]), 0, 100)
            barracks_count = min_max_scale(len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.Barracks and unit.build_progress == 100]),0, 8)
            barracks_building = min_max_scale(len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.Barracks and unit.build_progress < 100]), 0, 8)
            sd_count = min_max_scale(len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.SupplyDepot and unit.build_progress == 100]), 0, 10)
            sd_building = min_max_scale(len([unit for unit in obs.observation['feature_units'] if unit.unit_type == units.Terran.SupplyDepot and unit.build_progress < 100]), 0, 10)
            barracks_selected = int(self.unit_type_is_selected(obs, units.Terran.Barracks))
            scalar_observations = np.array([
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
            
            # Assuming obs is your current observation
            feature_screen = obs.observation['feature_screen']
            unit_density = np.array(feature_screen[features.SCREEN_FEATURES.unit_density.index])
            
            
            scalar_state = tf.convert_to_tensor([scalar_observations], dtype=tf.float32)
            spacial_state = tf.convert_to_tensor([unit_density], dtype=tf.float32)
            spacial_state = tf.expand_dims(spacial_state, axis=-1)
            

            
            action = self.choose_action(scalar_state, spacial_state)
            if action == 11:
                # hack fix as I couldn't find out what was causing key error 11, something to do with the distribuition
                action = 10
            if self.prev_action and self.train:
                self.learn(scalar_state, spacial_state, obs.reward - self.bonus, obs.last())
            self.prev_action = action
            self.prev_scalar_state = scalar_state
            self.prev_spacial_state = spacial_state
            self.punishment = 0
            if action in [3, 4, 6, 9]:
                x = random.randint(0,83)
                y = random.randint(0,83)
                return self.action_map[action](obs, (x,y))
            if action == 7:
                self.marine_training = True
            return self.action_map[action](obs)
        else:
            return actions.FUNCTIONS.no_op()
        
    def select_command(self, obs):
        commands = self.get_units_by_type(obs, units.Terran.CommandCenter)
        if len(commands) > 0:
            command = random.choice(commands)
            if command.x < 0:
                command.x = 0
            if command.y < 0:
                command.y = 0
            if command.x > 83:
                command.x = 83
            if command.y > 83:
                command.y = 83
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

    def move_scv(self, obs, position):
        if self.unit_type_is_selected(obs, units.Terran.SCV) and (actions.FUNCTIONS.Move_screen.id in obs.observation.available_actions):
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

    def move_marines(self, obs, position):
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

