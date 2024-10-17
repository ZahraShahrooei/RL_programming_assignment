import numpy as np
import gym

# Create environment
env = gym.make("CartPole-v1")

# Define bins for each of the four state variables
def create_bins(interval, num_bins):
    return np.linspace(interval[0], interval[1], num_bins)

# Define intervals and bins
intervals = [(-4.8, 4.8), (-2, 2), (-0.418, 0.418), (-2, 2)]
nbins = [15, 15, 15, 15]  # Number of bins for each state variable
bins = [create_bins(intervals[i], nbins[i]) for i in range(4)]

# Discretize the continuous state with clamping to ensure valid range
def discretize_bins(state):
    """Discretize continuous state into bins."""
    discretized_state = []
    for i in range(4):
        value = np.clip(state[i], bins[i][0], bins[i][-1])
        discretized_state.append(np.digitize(value, bins[i]) - 1)
    return tuple(discretized_state)

# Flatten the tuple state into a single index
def flatten_state(state, nbins):
    """Flatten a tuple state into a single integer index."""
    index = 0
    multiplier = 1
    for s, bin_size in zip(state, nbins):
        index += s * multiplier
        multiplier *= bin_size
    return index


num_states = np.prod(nbins)  # Total number of states = product of all bins
num_actions = env.action_space.n  # CartPole has two actions (left and right)

old_pi = np.ones((num_states, num_actions)) / num_actions
theta = 0.1
gamma = 1

# Policy Evaluation with deterministic transitions
def my_policy_evaluation(policy):
    V_previous = np.zeros(num_states)
    iteration_v = 0
    while True:
        Delta = 0
        iteration_v += 1
        V = np.zeros(num_states)

        # Loop over all discretized states
        for state_idx in range(num_states):
            expected_value = 0
            for action in range(num_actions):
                # Reverse mapping from index to tuple state
                env_state = env.reset()
                discretized_state = discretize_bins(env_state)
                state_idx = flatten_state(discretized_state, nbins)
                
                # Simulate transition for each action from this state
                env.state = np.array(discretized_state)  # Restore the state in env
                next_state, reward, done, _ = env.step(action)
                discretized_next_state = discretize_bins(next_state)
                next_state_idx = flatten_state(discretized_next_state, nbins)

                # If we reach a terminal state, no future value is added
                if done:
                    expected_value += policy[state_idx][action] * reward
                else:
                    expected_value += policy[state_idx][action] * (reward + gamma * V_previous[next_state_idx])

            V[state_idx] = expected_value
            Delta = max(Delta, np.abs(V[state_idx] - V_previous[state_idx]))

        if Delta < theta:
            break
        V_previous = np.copy(V)
    return V, iteration_v

# Policy Improvement
def my_policy_improvement(V):
    policy = np.zeros((num_states, num_actions))
    for state_idx in range(num_states):
        q_values = np.zeros(num_actions)
        for action in range(num_actions):
            env_state = env.reset()
            discretized_state = discretize_bins(env_state)
            state_idx = flatten_state(discretized_state, nbins)

            # Simulate transition
            env.state = np.array(discretized_state)  # Restore the state in env
            next_state, reward, done, _ = env.step(action)
            discretized_next_state = discretize_bins(next_state)
            next_state_idx = flatten_state(discretized_next_state, nbins)

            if done:
                q_values[action] = reward
            else:
                q_values[action] = reward + gamma * V[next_state_idx]

        optimal_action = np.argmax(q_values)
        policy[state_idx][optimal_action] = 1
    return policy

# Policy Iteration
def policy_iteration(policy):
    iteration = 0
    while True:
        iteration += 1
        V, _ = my_policy_evaluation(policy)
        new_policy = my_policy_improvement(V)

        if np.array_equal(new_policy, policy):
            break

        policy = np.copy(new_policy)
    return policy, iteration

MyOptimalPolicy, iteration = policy_iteration(old_pi)
print(MyOptimalPolicy)
print("Number of iterations:", iteration)

