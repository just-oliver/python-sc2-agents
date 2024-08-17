import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from collections import deque
import random

from absl import flags
import sys



FLAGS = flags.FLAGS
FLAGS(sys.argv)  # Initialize flags


class CollectMineralsAndGasEnv(gym.Env):
    def __init__(self):
        super(CollectMineralsAndGasEnv, self).__init__()
        self.previous_minerals = 0
        self.previous_gas = 0
        self.previous_workers = 0

        try:
            self.env = sc2_env.SC2Env(
                map_name="CollectMineralsAndGas",
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format=features.AgentInterfaceFormat(
                    feature_dimensions=features.Dimensions(screen=84, minimap=64),
                    use_feature_units=True),
                step_mul=16,
                game_steps_per_episode=0,
                visualize=True)
        except Exception as e:
            print(f"Error initializing SC2Env: {e}")
            self.env = None

        self.action_space = spaces.Discrete(len(actions.FUNCTIONS))
        self.observation_space = spaces.Box(low=0, high=4, shape=(42, 42, 1), dtype=np.uint8)
        self.available_actions = None

    @staticmethod
    def scale_reward(reward: float) -> float:
        shifted_reward = reward - 50
        scaled_reward = shifted_reward / 100
        return max(scaled_reward, 0.0)


    def calculate_reward(self, obs):
        reward = 0

        # Reward for collecting minerals
        mineral_difference = obs.observation.player.minerals - self.previous_minerals
        reward += mineral_difference * 1  # 1 point per mineral collected

        # Reward for collecting gas
        gas_difference = obs.observation.player.vespene - self.previous_gas
        reward += gas_difference * 2  # 2 points per gas collected

        # Reward for creating workers
        worker_difference = obs.observation.player.food_workers - self.previous_workers
        reward += worker_difference * 10  # 10 points per new worker

        # Penalty for idle workers
        idle_worker_count = obs.observation.player.idle_worker_count
        reward -= idle_worker_count * 0.1  # -0.1 points per idle worker

        # Small negative reward for each step to encourage faster completion
        reward -= 0.1

        return self.scale_reward(reward)
    def reset(self):
        if self.env is None:
            raise RuntimeError("SC2Env not properly initialized")
        obs = self.env.reset()[0]
        self.available_actions = obs.observation.available_actions
        self.previous_minerals = 0
        self.previous_gas = 0
        self.previous_workers = 0
        return self._process_state(obs)

    def _process_state(self, obs):
        # Downsampling the state to 42x42
        state = obs.observation.feature_screen.player_relative
        return np.array(state[::2, ::2, np.newaxis], dtype=np.uint8)

    def step(self, action):
        if self.env is None:
            raise RuntimeError("SC2Env not properly initialized")

        if action not in self.available_actions:
            action = np.random.choice(self.available_actions)

        args = []
        for arg in actions.FUNCTIONS[action].args:
            if arg.name in ['screen', 'screen2', 'minimap']:
                args.append([np.random.randint(0, 42), np.random.randint(0, 42)])
            elif arg.name == 'queued':
                args.append([0])
            else:
                args.append([np.random.randint(0, size) if size > 0 else 0 for size in arg.sizes])

        obs = self.env.step([actions.FunctionCall(action, args)])[0]
        state = self._process_state(obs)
        reward = self.calculate_reward(obs)
        done = obs.last()
        score = obs.observation.score_cumulative[0]

        self.available_actions = obs.observation.available_actions

        return state, reward, done, {'score': score}

    def close(self):
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()
# Define the Deep Q-Network
class DQN(tf.keras.Model):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(action_size, activation='linear')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


tf.config.run_functions_eagerly(True)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.frame_skip = 8  # Only predict every 8th frame
        self.update_frequency = 1000  # Update the model every 1000 steps

    def _build_model(self):
        model = Sequential([
            Conv2D(16, (4, 4), strides=(2, 2), activation='relu', input_shape=self.state_size),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, available_actions):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(available_actions)
        act_values = self.model.predict(state[np.newaxis, ...], verbose=0)[0]
        available_act_values = [act_values[i] for i in available_actions]
        return available_actions[np.argmax(available_act_values)]

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state[np.newaxis, ...], verbose=0)[0])
            target_f = self.model.predict(state[np.newaxis, ...], verbose=0)
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f[0])

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0, batch_size=batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
# Main training loop
def main():
    try:
        env = CollectMineralsAndGasEnv()
        if env.env is None:
            print("Failed to initialize the environment. Exiting.")
            return

        state_size = env.observation_space.shape
        action_size = env.action_space.n
        agent = DQNAgent(state_size, action_size)
        batch_size = 32
        episodes = 1000
        total_steps = 0

        for e in range(episodes):
            state = env.reset()
            total_score = 0
            total_reward = 0
            for time in range(1000):  # Increase max steps per episode
                if env.available_actions is None or len(env.available_actions) == 0:
                    print("No available actions. Resetting environment.")
                    break

                if time % agent.frame_skip == 0:
                    action = agent.act(state, env.available_actions)

                next_state, reward, done, info = env.step(action)
                total_reward += reward
                total_score += info.get('score', 0)  # Assuming the environment returns a score in info

                if time % agent.frame_skip == 0:
                    agent.remember(state, action, reward, next_state, done)
                    state = next_state

                total_steps += 1

                if done:
                    break

                if len(agent.memory) > batch_size and total_steps % agent.update_frequency == 0:
                    agent.replay(batch_size)

            print(f"Episode: {e}/{episodes}, Score: {total_score}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2}")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'env' in locals():
            env.close()


if __name__ == "__main__":
    main()