import numpy as np
import tensorflow as tf
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


class SelfDrivingCarEnv:
    def __init__(self):
        self.action_space = 4  # forward, left, right, stop
        self.observation_space = 2  # position (x, y)
        self.destination = np.array([10, 10])  # Destination coordinates
        self.position = np.array([0, 0])  # Initial position
        self.max_steps = 100  # Maximum number of steps
        self.current_step = 0  # Current step

    def reset(self):
        self.position = np.array([0, 0])  # Reset current state of position
        self.current_step = 0  # Reset step count to  the
        return self.position

    def step(self, action):
        # Move according to action
        if action == 0:  # Forward
            self.position[1] += 1
        elif action == 1:  # Left
            self.position[0] -= 1
        elif action == 2:  # Right
            self.position[0] += 1
        elif action == 3:  # Stop
            pass

        # Calculate reward
        distance_to_destination = np.linalg.norm(self.destination - self.position)
        reward = -distance_to_destination  # Negative distance as reward

        # Check if reached destination
        if np.allclose(self.position, self.destination):
            done = True
            reward = 100  # Reward for reaching destination
        else:
            done = False

        # Penalize if out of bounds
        if (self.position < 0).any() or (self.position > self.destination).any():
            done = True
            reward = -100  # Penalize for going out of bounds

        # Increment step count
        self.current_step += 1

        # Check if maximum steps reached
        if self.current_step >= self.max_steps:
            done = True

        return self.position, reward, done, {}


# Create environment
env = SelfDrivingCarEnv()


# Build a neural network model
def build_model(input_shape, num_actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + input_shape))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_actions, activation='linear'))
    return model


# Define the parameters for the DQN agent
input_shape = env.observation_space
num_actions = env.action_space
model = build_model(input_shape, num_actions)

# Build the DQN agent
memory = SequentialMemory(limit=1000, window_length=1)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=num_actions,
               enable_double_dqn=True, enable_dueling_network=True)
dqn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), metrics=['mae'])

# Train the DQN agent
dqn.fit(env, nb_steps=1000, visualize=False, verbose=1)

# Test the DQN agent
dqn.test(env, nb_episodes=5, visualize=True)

# Save the DQN agent
dqn.save_weights('dqn_weights.h5', overwrite=True)

# Close the environment
env.close()
