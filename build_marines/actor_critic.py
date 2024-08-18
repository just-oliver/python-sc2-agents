#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Oliver Just-De Bleser
"""

import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Concatenate, Flatten

class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=64, fc2_dims=64,
                 name='actor_critic', chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        
        
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ac.weights.h5')
        os.make(self.checkpoint_dir, exist_ok=True)
        
        # spacial data layers
        self.conv1 = Conv2D(32, (8, 8), strides=4, activation='relu', padding='same', input_shape=(1,84,84,10))
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv2 = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.conv3 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')
        self.pool3 = MaxPooling2D(pool_size=(2, 2))
        # flatten spacial features for Dense layers
        self.flatten = Flatten()
        # shared layers
        self.concat = Concatenate()
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        # positional outputs
        self.x_pos = Dense(1, activation='sigmoid')
        self.y_pos = Dense(1, activation='sigmoid')
        # value output
        self.v = Dense(1, activation=None)
        # policy output
        self.pi = Dense(n_actions, activation='softmax')
        

    def call(self, scalar_observations, spacial_observations):
        conv_1 = self.conv1(spacial_observations)
        pool_1 = self.pool1(conv_1)
        conv_2 = self.conv2(pool_1)
        pool_2 = self.pool2(conv_2)
        conv_3 = self.conv3(pool_2)
        pool_3 = self.pool3(conv_3)
        flatten = self.flatten(pool_3)
        # shared layers
        merged = self.concat([scalar_observations, flatten])
        fc_1 = self.fc1(merged)
        fc_2 = self.fc2(fc_1)
        # position output
        x_pos = self.x_pos(fc_2)
        y_pos = self.y_pos(fc_2)
        # value output
        v = self.v(fc_2)
        # policy output
        pi = self.pi(fc_2)
        return v, pi, x_pos, y_pos

class ActorCriticNetwork_nopos(keras.Model):
    def __init__(self, n_actions, fc1_dims=64, fc2_dims=64,
                 name='actor_critic', chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork_nopos, self).__init__()
        
        
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_ac.weights.h5')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # spacial data layers
        # self.conv1 = Conv2D(32, (8, 8), strides=4, activation='relu', padding='same', input_shape=(84,84,1))
        # self.pool1 = MaxPooling2D(pool_size=(2, 2))
        # self.conv2 = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')
        # self.pool2 = MaxPooling2D(pool_size=(2, 2))
        # self.conv3 = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')
        # self.pool3 = MaxPooling2D(pool_size=(2, 2))
        # # flatten spacial features for Dense layers
        # self.flatten = Flatten()
        # # shared layers
        # self.concat = Concatenate()
        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        # positional outputs
        # value output
        self.v = Dense(1, activation=None)
        # policy output
        self.pi = Dense(n_actions, activation='softmax')
        

    def call(self, scalar_observations, spacial_observations):
        # conv_1 = self.conv1(spacial_observations)
        # pool_1 = self.pool1(conv_1)
        # conv_2 = self.conv2(pool_1)
        # pool_2 = self.pool2(conv_2)
        # conv_3 = self.conv3(pool_2)
        # pool_3 = self.pool3(conv_3)
        # flatten = self.flatten(pool_3)
        # # shared layers
        # merged = self.concat([scalar_observations, flatten])
        fc_1 = self.fc1(scalar_observations)
        fc_2 = self.fc2(fc_1)
        # value output
        v = self.v(fc_2)
        # policy output
        pi = self.pi(fc_2)
        return v, pi