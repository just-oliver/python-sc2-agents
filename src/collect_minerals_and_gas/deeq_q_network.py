import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(self._calculate_conv_output(input_shape), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _calculate_conv_output(self, input_shape):
        o = self.conv1(torch.zeros(1, *input_shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, input_shape, n_actions, gamma=0.99, lr=0.0001, epsilon_start=1.0, epsilon_final=0.01, epsilon_decay=50000, replay_size=10000, batch_size=32):
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.replay_size = replay_size
        self.batch_size = batch_size

        self.model = DQN(input_shape, n_actions)
        self.target_model = DQN(input_shape, n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = deque(maxlen=self.replay_size)

    def update_epsilon(self, step):
        self.epsilon = self.epsilon_final + (1.0 - self.epsilon_final) * np.exp(-step / self.epsilon_decay)

    def select_action(self, state, step):
        self.update_epsilon(step)
        if random.random() < self.epsilon:
            return random.randrange(self.model.fc2.out_features)
        else:
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0)
            q_values = self.model(state)
            return q_values.max(1)[1].item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.float32(states))
        next_states = torch.FloatTensor(np.float32(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        next_q_values = self.target_model(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        expected_q_value = rewards + self.gamma * next_q_value * (1 - dones)

        loss = F.mse_loss(q_value, expected_q_value.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())