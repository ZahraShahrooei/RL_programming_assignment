
# !pip install gymnasium
# !pip install pot

from __future__ import annotations
import matplotlib.pyplot as plt

import numpy as np
import gymnasium as gym
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict

env = gym.make("CartPole-v1")

def create_bins(i,num):
    return np.arange(num+1)*(i[1]-i[0])/num+i[0]


ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
nbins = [50,20,5,20] # number of bins for each parameter
bins = [create_bins(ints[i],nbins[i]) for i in range(4)]

def discretize_bins(x):
    return tuple(np.digitize(x[i],bins[i]) for i in range(4))

num_states = np.prod(nbins)
nb_actions = env.action_space.n

eps = 0.1  # exploration probability
decay_ep = 0.999  # decay factor for epsilon
alpha = 0.2  # learning rate
gamma = 0.99  # discount factor
min_eps = 0.001  # minimum exploration probability

### function: generate_episode
# inputs:
#   - `env`: the environment to interact with
#   - `policy`: a function to select an action based on the current state and policy probabilities
#   - `pi`: a dictionary representing the current policy (action probabilities for each state-action pair)
# outputs:
#   - `trajectory`: a list of tuples, each containing the state, reward, done flag, and action for each step of the episode
#   - `len(trajectory) - 1`: the total number of steps in the episode
# process:
#   - The agent interacts with the environment using the provided `policy`.
#   - At each step, the agent selects an action based on the current state and the policy `pi`, and the environment returns the next state and reward.
#   - The function tracks the sequence of states, actions, and rewards (the trajectory) until the episode ends (`done` is True).

def generate_episode(env, policy,pi):
  done=True
  trajectory=[]
  while True:
    if done:
      # print(env.reset())
      # St,Rt, done=env.reset(), None, False
      St, info = env.reset()
      St = discretize_bins(St)
      Rt, done = None, False

    else:
      St, Rt, done,_,_=env.step(At)
      St = discretize_bins(St)

      # print(St)
    # St = tuple(St)
    At=policy(St,pi)
    trajectory.append((St, Rt,done, At))
    if done:
      break
  return trajectory, len(trajectory)-1


### function: argmax_rand
# inputs:
#   - `arr`: an array of values (e.g., Q-values for a state-action pair)
# outputs:
#   - the index of the maximum value in `arr`, randomly selected if there are multiple maximum values
# process:
#   - The function identifies all elements in `arr` that are equal to the maximum value and randomly selects one of them.

def argmax_rand(arr):

  return np.random.choice(np.flatnonzero(arr==np.max(arr)))

### function: on_policy_mc_control
# inputs:
#   - `env`: the environment in which the agent interacts (e.g., CartPole)
#   - `ep`: the number of episodes to run the algorithm for
#   - `gamma`: the discount factor for future rewards
#   - `eps`: the exploration probability for epsilon-greedy policy
# outputs:
#   - `Q`: a dictionary containing the estimated Q-values for each state-action pair
#   - `pi`: the updated policy as a dictionary mapping state-action pairs to action probabilities
#   - `return_ep`: a list of discounted returns for each episode
# process:
#   - The algorithm iterates over `ep` episodes, generating a trajectory in each episode using the current policy.
#   - For each state-action pair in the trajectory, the discounted return is calculated, and the Q-values are updated using the average of the returns.
#   - The policy is updated after each episode using an epsilon-greedy approach, where the action with the highest Q-value is favored, but exploration is still possible with probability `eps`.


def on_policy_mc_control(env, ep, gamma, eps):
    def policy(St, pi):
        return np.random.choice([0, 1], p=[pi[(St, a)] for a in [0, 1]])

    pi = defaultdict(lambda: 1 / nb_actions)
    Q = defaultdict(float)
    Returns = defaultdict(list)

    return_ep = []  # List to track discounted return per episode

    for m in range(ep):
        traj, T = generate_episode(env, policy, pi)
        G = 0  # Discounted return
        for t in range(T - 1, -1, -1):
            St, _, _, At = traj[t]
            _, Rt_1, _, _ = traj[t + 1]
            G = gamma * G + Rt_1  # Discounted return calculation
            if (St, At) not in [(traj[i][0], traj[i][3]) for i in range(0, t)]:
                Returns[(St, At)].append(G)
                Q[(St, At)] = np.mean(Returns[(St, At)])
                A_star = argmax_rand([Q[(St, a)] for a in range(nb_actions)])
                for a in range(nb_actions):
                    if a == A_star:
                        pi[(St, a)] = 1 - eps + eps / nb_actions
                    else:
                        pi[(St, a)] = eps / nb_actions

        # Append the discounted return of the episode to return_ep
        return_ep.append(G)

        if m % 10 == 0:
            eps = 0.99 * eps

    return Q, pi, return_ep

# Run the on-policy Monte Carlo control algorithm
Q, pi, return_ep = on_policy_mc_control(env, ep=100000, gamma=1, eps=1)

# Plotting the return over episodes
plt.figure(figsize=(10, 6))
plt.plot(return_ep)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.xlabel('Episode')
plt.ylabel('Total Return')
plt.xlim([0, 100000])
plt.ylim([0, 500])
plt.grid(True)
plt.show()


# Function to calculate moving average
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Plotting the raw return over episodes
plt.figure(figsize=(10, 6))
plt.plot(return_ep,alpha=0.3, label='Total Return')
window_size = 100 
return_ep_ma = moving_average(return_ep, window_size)
plt.plot(return_ep, alpha=0.3,color='red', label="Original Returns")
plt.plot(range(window_size-1, len(return_ep)), return_ep_ma, label=f'Moving Average (window={window_size})', color='red')
# Plot settings
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.xlabel('Episode')
plt.ylabel('Total Return')
plt.xlim([0, 100000])
plt.ylim([0, 500])
plt.grid(True)
plt.legend()
plt.show()