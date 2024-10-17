
# !pip install gymnasium
# !pip install matplotlib

from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib as mpl
import logging
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import gymnasium as gym
from scipy.linalg import eig
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

env = gym.make("CartPole-v1")
# print(env.action_space)
# print(env.observation_space)
# print(env.action_space.sample())

###----Q-learning------###
from collections import defaultdict

### function: create_bins
# inputs: an interval `i` (tuple) and `num` (number of bins)
# outputs: an array of bin edges
# process: Creates evenly spaced bins within the interval for discretization.
def create_bins(i,num):
    return np.arange(num+1)*(i[1]-i[0])/num+i[0]


ints = [(-5,5),(-2,2),(-0.5,0.5),(-2,2)] # intervals of values for each parameter
nbins = [50,20,5,20] # number of bins for each parameter
bins = [create_bins(ints[i],nbins[i]) for i in range(4)]

### function: discretize_bins
# inputs: a state `x` (list of 4 continuous values)
# outputs: a tuple representing the discretized state
# process: Maps each continuous value to the corresponding bin index.
def discretize_bins(x):
    return tuple(np.digitize(x[i],bins[i]) for i in range(4))

num_states = np.prod(nbins)
nb_actions = env.action_space.n

eps=0.1
decay_ep=0.999
alpha=0.2
gamma=0.99
min_eps=0.001
# nb_actions=2

### function: argmax_rand
# inputs: an array `arr` of action values
# outputs: an index of the action with the highest value, breaking ties randomly
# process: Selects the index of the maximum value from the array, with random tie-breaking.
def argmax_rand(arr):
  return np.random.choice(np.flatnonzero(arr == np.max(arr)))

### function: SARSA
# inputs: exploration rate `eps`, learning rate `alpha`, number of episodes `ep` (default 1000)
# outputs: Q-values `Q`, policy `pi`, and the return over episodes `return_ep`
# process: Implements the SARSA algorithm, updates Q-values and the policy using an epsilon-greedy approach.
def q_learning(eps, alpha,ep=1000):
  Q=defaultdict(float)  ##

  ### function: policy
  # inputs: a state `s` and policy `pi`
  # outputs: an action based on the policy distribution
  # process: Chooses an action according to the given policy.
  def policy(s,pi):
     return np.random.choice([0, 1], p=[pi[(s, a)] for a in range(nb_actions)])

  pi=defaultdict(lambda: 1 / nb_actions)

  return_ep=[]
  for _ in range(ep):
    total_return=0

    s = discretize_bins(env.reset()[0])


    while True:
      a=policy(s,pi)
      s_,r,done,_,_=env.step(a)
      # print(f"before discretization: {s_}")
      s_ = discretize_bins(s_)
      # print(f"after discretization: {s_}")
      total_return = total_return + r
      q_max=max([Q[(s_,at)] for at in range(nb_actions) ])
      Q[(s,a)]= Q[(s,a)] + alpha * (r + gamma *q_max - Q[(s,a)])
      ###policy improvement phase
      A_star=argmax_rand([Q[(s,a)] for a in range(nb_actions)])
      for a in range(nb_actions):
        if a==A_star:
          pi[(s,a)]=1 - eps + eps/nb_actions
        else:
          pi[(s,a)]=eps/nb_actions

      s=s_
      if done:
        break
    return_ep.append(total_return)
    eps = max(min_eps, eps * decay_ep)
  return Q, pi, return_ep

Q, pi, return_ep =q_learning(eps, alpha,ep=10000)
pi = pd.DataFrame(pi.items(),columns=["state-action","probability"])
# print(pi.head(20))

# Plotting the return over episodes
plt.figure(figsize=(10, 6))
plt.plot(return_ep)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.xlabel('Episode')
plt.ylabel('Total Return')
plt.xlim([0, 10000])
plt.ylim([0, 200])
plt.grid(True)
plt.show()
import numpy as np

### function: moving_average
# inputs:
#   - `data`: a list or array of data points (in this case, total returns per episode)
#   - `window_size`: an integer representing the number of consecutive episodes to average over
# outputs:
#   - an array of smoothed data (moving average of the input data)
# process:
#   - Uses `np.convolve` to compute the moving average of the data over a specified window size.
#   - A convolution operation is applied between the `data` and an array of `1/window_size` to average over the window.
#   - The mode `'valid'` ensures that only data points where the window fully fits are included in the result, shortening the output array.
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


smoothed_return_ep = moving_average(return_ep, 100)
plt.figure(figsize=(10, 6))
plt.plot(return_ep, alpha=0.3,color='red', label="Original Returns")
plt.plot(range(len(smoothed_return_ep)), smoothed_return_ep, color='red', label="Smoothed Returns (Window Size = 100)")

plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(1.5)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.xlabel('Episode')
plt.ylabel('Total Return')
plt.xlim([0, 10000])
plt.ylim([0, 200])
plt.grid(True)
plt.legend()
plt.show()