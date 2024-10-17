import gym
import numpy as np
import matplotlib.pyplot as plt

# Parameters
NUM_EPISODES = 20000
MAX_T =499
gamma = 0.999  # Discount factor
ALPHA = 0.00001  # Learning rate
EPSILON = 0.1  # Initial exploration rate
EPSILON_DECAY = 0.9995  # Exploration decay rate

# Create the environment
env = gym.make('CartPole-v0')
NUM_ACTIONS = env.action_space.n
NUM_OBS = env.observation_space.shape[0]

# Initialize weights for the linear model (Q-function approximation) with 10 features
weights = np.random.uniform(low=-0.01, high=0.01, size=(10, NUM_ACTIONS))

# Feature extraction function
def feature_extractor(state):
    # Normalization factors for each feature to ensure consistency
    norms = np.array([2.4, 4, np.radians(12), 5, 2.4**2, 4**2, np.radians(12)**2, 5**2, 2.4*np.radians(12), 4*5])

    # Extract different features from the state
    features = np.array([
        state[0],       # Cart position
        state[1],       # Cart velocity
        state[2],       # Pole angle
        state[3],       # Pole angular velocity
        state[0] ** 2,  # Square of cart position
        state[1] ** 2,  # Square of cart velocity
        state[2] ** 2,  # Square of pole angle
        state[3] ** 2,  # Square of pole angular velocity
        state[0] * state[2],  # Interaction term: cart position * pole angle
        state[1] * state[3]   # Interaction term: cart velocity * pole angular velocity
    ])

    # Normalize the features
    normalized_features = features / norms
    return normalized_features

def epsilon_greedy_action(state, epsilon):
    """Select action using epsilon-greedy strategy."""
    features = feature_extractor(state)  # Use extracted features
    if np.random.rand() < epsilon:
        return env.action_space.sample()  # Random action
    else:
        q_values = np.dot(features, weights)  # Linear Q-function approximation with features
        return np.argmax(q_values)

def update_weights(state, action, reward, next_state, done):
    """Perform the Q-learning update step."""
    global weights
    features = feature_extractor(state)
    next_features = feature_extractor(next_state)

    q_values = np.dot(features, weights)
    q_next_values = np.dot(next_features, weights)

    # Target for Q-learning
    target = reward if done else reward + gamma * np.max(q_next_values)

    # Q-learning update rule
    q_values[action] += ALPHA * (target - q_values[action])

    # Update the weights
    weights += ALPHA * np.outer(features, q_values)

# Function to calculate moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Training loop
rewards = [] 
for episode in range(NUM_EPISODES):
    state = env.reset()  
    total_reward = 0
    epsilon = max(0.01, EPSILON * (EPSILON_DECAY ** episode))  # Decaying epsilon

    for t in range(MAX_T):
        action = epsilon_greedy_action(state, epsilon)
        next_state, reward, done,_ = env.step(action)

        # Update weights using Q-learning
        update_weights(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        if done:
            break

    rewards.append(total_reward)  # Record the total reward for the episode

    # Print progress every 100 episodes
    # if episode % 100 == 0:
    #   pass
        # print(f"Episode {episode} - Total reward: {total_reward}, Epsilon: {epsilon:.2f}")

    # Check for early stopping condition
    # if total_reward >= 475:
    #     print(f"Solved in episode {episode}")
    #     break

print("Training completed.")

###---stability
print(np.std(rewards))

# Plotting both the raw return and moving average return
plt.figure(figsize=(10, 6))
plt.plot(range(len(rewards)), rewards, alpha=0.3, label='Total Return')
window_size = 50
if len(rewards) >= window_size:
    rewards_ma = moving_average(rewards, window_size)
    # Plot the moving average return
    plt.plot(range(window_size - 1, len(rewards)), rewards_ma, label=f'Moving Average (window={window_size})', color='red')

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
# plt.title('Total Rewards and Moving Average per Episode')
plt.legend()
plt.show()
