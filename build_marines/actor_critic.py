import os
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D

class ActorCriticNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512,
                 name='actor_critic', chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')

        # spacial data layers
        self.conv1 = Conv2D(32, (8, 8), strides=4, activation='relu')
        self.pool1 = MaxPooling2D(pool_size=(2, 2))

        self.conv2 = Conv2D(64, (4, 4), strides=2, activation='relu')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))

        self.conv3 = Conv2D(64, (3, 3), strides=1, activation='relu')
        self.pool3 = MaxPooling2D(pool_size=(2, 2))


        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.x_pos = Dense(84, activation='softmax')
        self.y_pos = Dense(84, activation='softmax')
        self.v = Dense(1, activation=None)
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)
        v = self.v(value)
        pi = self.pi(value)
