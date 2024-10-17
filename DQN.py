
# !pip install gymnasium
# !pip install matplotlib
# !pip install tensorflow
from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import random
from collections import deque
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
env = gym.make("CartPole-v1")
M = 500  # episodes
T = 210  # steps
batch_size = 24


class DQN():
    def __init__(self, states, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay, tau=0.1):
        """
        Input:
        - states: Number of state features (observation space).
        - actions: Number of possible actions (action space).
        - alpha: Learning rate.
        - gamma: Discount factor for future rewards.
        - epsilon: Initial exploration rate for the epsilon-greedy policy.
        - epsilon_min: Minimum exploration rate.
        - epsilon_decay: Decay rate for epsilon after each episode.
        """
        
        self.gamma = gamma
        self.memory = deque([], maxlen=2500)  # for replay buffer
        self.nS = states
        self.nA = actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau  # Parameter for soft updating target network
        self.model = self.build_model()  # Main Q-Network
        self.target_model = self.build_model()  # Target Q-Network
        self.update_target_network()  # Initialize target network with same weights
        self.loss = []
        
    #  Input: None (uses parameters from the class)
    # Output: Returns a compiled Keras model.
    # - Builds a Sequential neural network with two hidden layers of 24 units each (with ReLU activation).
    # - The output layer has the same number of units as the number of actions (linear activation) because this is a regression problem.
    # - The model is compiled with Adam optimizer and mean squared error loss function.

    def build_model(self):
        model = models.Sequential([
            layers.Dense(24, activation='relu', input_dim=self.nS),
            layers.Dense(24, activation='relu'),
            layers.Dense(self.nA, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.alpha), loss="mse")
        return model

    def update_target_network(self):
        """Copy the weights from the main model to the target model"""
        self.target_model.set_weights(self.model.get_weights())
    
    
    
    #Input: None (uses the replay buffer to fetch minibatches)
    # Output: Updates the model's weights based on the training data.
    # - Samples a minibatch from the memory buffer.
    # - For each sample in the minibatch, it predicts the current Q-values for both the current and next state.
    # - Uses the Bellman equation to compute the target Q-values for the next state.
    # - Fits the model on the computed target Q-values.
    # - Appends the loss history for analysis and updates epsilon for exploration-exploitation trade-off.

    def train(self):
        minibatch = random.sample(self.memory, batch_size)
        x = []
        y = []
        st = np.zeros((0, self.nS))
        nst = np.zeros((0, self.nS))
        for i in range(len(minibatch)):
            st = np.append(st, minibatch[i][0], axis=0)
            nst = np.append(nst, minibatch[i][3], axis=0)

        st_predict = self.model.predict(st)
        nst_predict = self.target_model.predict(nst)  # Use target network for next state prediction
        index = 0
        for state, action, reward, nstate, done in minibatch:
            x.append(state)
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(nst_predict[index])
            target_f = st_predict[index]
            target_f[action] = target
            y.append(target_f)
            index += 1

        x = np.array(x).reshape(batch_size, self.nS)
        y = np.array(y)
        hist = self.model.fit(x, y, epochs=1, verbose=0)
        self.loss.append(hist.history['loss'][0])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def epsilon_greedy(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.nA)
        else:
            action_values = self.model.predict(state)[0]  # Q(s, 0) and Q(s,1)
            return np.argmax(action_values)
    # Input:
    # - state: The current state of the environment.
    # - action: The action taken in the current state.
    # - reward: The reward received after taking the action.
    # - next_state: The next state the environment transitions to.
    # - done: Whether the episode has ended.
    # Output: None (stores the experience in memory).
    # How the function processes:
    # - Stores the transition (state, action, reward, next_state, done) in the replay memory buffer.

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


nS = env.observation_space.shape[0]
nA = env.action_space.n
alpha = 0.001
gamma = 1
epsilon = 0.1
epsilon_min = 0.01
epsilon_decay = 0.995

dqn = DQN(nS, nA, alpha, gamma, epsilon, epsilon_min, epsilon_decay)

# List to store total rewards for each episode
rewards = []
# Input: None (manages the environment interactions and DQN training for each episode)

# Output: Updates the total rewards for each episode and trains the DQN model.

# How the function processes:
# - Resets the environment and runs the agent for each step in an episode.
# - The agent selects actions using epsilon-greedy policy.
# - The transition (state, action, reward, next_state, done) is stored in memory.
# - After every step, the model is trained using the replay buffer.
# - At the end of the episode, the reward is tracked for performance analysis.


for episode in range(M):
    state = env.reset()
    state = np.reshape(state, [1, dqn.nS])  # NN input must be 2-dimensional
    total_reward = 0
    for t in range(T):
        action = dqn.epsilon_greedy(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, dqn.nS])

        total_reward += reward
        dqn.store(state, action, reward, next_state, done)
        if len(dqn.memory) > batch_size:
            dqn.train()

        state = next_state
        # env.render()  # Optional, can be disabled
        if done:
            print(f"Episode: {episode}, Score: {total_reward}, Epsilon: {dqn.epsilon}")
            break

    rewards.append(total_reward)  # Track total rewards

    # Update the target network every 10 episodes
    if episode % 10 == 0:
        dqn.update_target_network()

# Plotting the learning curve
plt.plot(range(M), rewards)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.show()