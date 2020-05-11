import numpy as np
import matplotlib.pyplot as plt
import time
import random

import os
from tqdm import tqdm

class RandomWalkEnvironment():
    def env_init(self, env_info={}):
        # env_info dict contains: num_states, start_state, left_terminal_state, right_terminal_state, seed

        # self.rand_generator = np.random.RandomState(env_info.get("seed")) # use for repeatability
        self.rand_generator = np.random.RandomState(round(time.clock()*1000000)) # use for randomness
        
        self.num_states = env_info["num_states"]
        self.start_state = env_info["start_state"]
        self.left_terminal_state = env_info["left_terminal_state"]
        self.right_terminal_state = env_info["right_terminal_state"]

    def env_start(self):
        
        reward = 0.0
        state = self.start_state
        is_terminal = False

        self.reward_state_term = (reward, state, is_terminal)

        return self.reward_state_term[1] # returning first state from the env

    def env_step(self, action):

        last_state = self.reward_state_term[1]
        print("last state: {0}".format(last_state))

        # all transactions beyond terminal states are absorbed into terminal state
        if action == 0: #left
            current_state = max(self.left_terminal_state, last_state + self.rand_generator.choice(range(-100,0)))
        elif action == 1: #right
            current_state = min(self.right_terminal_state, last_state + self.rand_generator.choice(range(1,101)))
        else:
            raise ValueError("Wrong action value")

        # terminate left
        if current_state == self.left_terminal_state:
            reward = -1.0
            is_terminal = True      
        elif current_state == self.right_terminal_state:
            reward = 1.0
            is_terminal = True
        else:
            reward = 0.0
            is_terminal = False

        self.reward_state_term = (reward, current_state, is_terminal)

        return self.reward_state_term



############# agent
def agent_policy(rand_generator, state):
    chosen_action = rand_generator.choice([0,1])
    return chosen_action

# here we apply state aggregation to reduce the states, so we group closer states together by one-hot encoding
def get_state_feature(num_states_in_feature, num_features, state):
    one_hot_vector = np.zeros(num_features)
    one_hot_vector[ (state-1) // num_states_in_feature ] = 1
    return one_hot_vector

class TDAgent():
    def __init__(self):
        self.num_states = None
        self.num_features = None
        self.step_size = None
        self.discount_factor = None

    def agent_init(self, agent_info={}):
        # agent_info dict contains: num_states, num_features, step_size, discount_factor, seed

        self.rand_generator = np.random.RandomState(agent_info.get("seed"))

        self.num_states = agent_info.get("num_states")
        self.num_features = agent_info.get("num_features")
        self.step_size = agent_info.get("step_size")
        self.discount_factor = agent_info.get("discount_factor")

        num_states_in_feature = int(self.num_states/self.num_features)

        # we can map states in the higher dimension to lower dimension beforehand to save time during training 
        self.all_state_features = np.array([get_state_feature(num_states_in_feature, self.num_features, state) for state in range(1, self.num_states+1)])

        self.weights = np.zeros(self.num_features)

        self.last_state = None
        self.last_action = None

    def agent_step(self, reward, state):

        last_state_feature = self.all_state_features[self.last_state-1]

    def agent_start(self, state):
        self.last_state = state
        self.last_action = agent_policy(self.rand_generator, state)
        return self.last_action


########### test area
current_env = RandomWalkEnvironment()
environment_parameters = {
    "num_states" : 500, 
    "start_state" : 250,
    "left_terminal_state" : 0,
    "right_terminal_state" : 501, 
    "seed" : 1,
    "discount_factor" : 1.0
}
current_env.env_init(environment_parameters)
current_env.env_start()

is_terminal = False
ctr = 0
while not is_terminal:
    current_action = random.randint(0,1)
    trio = current_env.env_step(current_action)
    ctr = ctr + 1
    is_terminal = trio[2]

print("ctr={0}".format(ctr))