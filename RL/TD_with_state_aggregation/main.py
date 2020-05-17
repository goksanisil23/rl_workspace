import numpy as np
import matplotlib.pyplot as plt
import time
import random

import os
from tqdm import tqdm

#################### Random Walk Environment implementation ########################### 
class RandomWalkEnvironment():
    def env_init(self, env_info={}):
        # env_info dict contains: num_states, start_state, left_terminal_state, right_terminal_state, seed

        self.rand_generator = np.random.RandomState(env_info.get("seed")) # use for repeatability
        # self.rand_generator = np.random.RandomState(round(time.clock()*1000000)) # use for randomness
        
        self.num_states = env_info["num_states"]
        self.start_state = env_info["start_state"]
        self.left_terminal_state = env_info["left_terminal_state"]
        self.right_terminal_state = env_info["right_terminal_state"]

        self.true_state_values = np.linspace(-1, 1, self.num_states)

    def env_start(self):
        
        reward = 0.0
        state = self.start_state
        is_terminal = False

        self.reward_state_term = (reward, state, is_terminal)

        return self.reward_state_term[1] # returning first state from the env

    def env_step(self, action):

        last_state = self.reward_state_term[1]
        # print("last state: {0}".format(last_state))

        # all transactions beyond terminal states are absorbed into terminal state
        # stepping randomly with leaps os (0,100) to either right or left
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



############# Agent ####################################
def agent_policy(rand_generator, state):
    # agent randomly chooses to go right or left with equal probability
    chosen_action = rand_generator.choice([0,1])
    return chosen_action

# here we apply state aggregation to reduce the states, so we group closer states together by one-hot encoding to form features
def get_state_feature(num_states_in_feature, num_features, state):
    one_hot_vector = np.zeros(num_features)
    one_hot_vector[ (state-1) // num_states_in_feature ] = 1 # note: states start from index 1
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
        print(self.all_state_features.shape)

        self.weights = np.zeros(self.num_features)

        self.last_state = None
        self.last_action = None

    def agent_start(self, state):
        self.last_state = state
        self.last_action = agent_policy(self.rand_generator, state)
        return self.last_action


    def agent_step(self, reward, state):

        last_state_feature = self.all_state_features[self.last_state-1]
        current_state_feature = self.all_state_features[state-1]

        # state value calculation with linear func approximation (v = w*x^T)
        last_value = np.dot(last_state_feature, self.weights)
        value = np.dot(current_state_feature, self.weights)

        # semi-gradient TD(0) Weight Update
        self.weights += self.step_size * (reward + self.discount_factor*value - last_value)*last_state_feature # gradient of value function in lineer func becomes the feature vector

        self.last_state = state
        self.last_action = agent_policy(self.rand_generator, state)
        return self.last_action

    def agent_end(self, reward): # runs when agent terminate
        last_state_feature = self.all_state_features[self.last_state-1]
        # state value calculation with linear func approximation (v = w*x^T)
        last_value = np.dot(last_state_feature, self.weights)        
        # semi-gradient TD(0) Weight Update (the value of the terminal state is 0)
        self.weights += self.step_size * (reward + self.discount_factor*0 - last_value)*last_state_feature # gradient of value function in lineer func becomes the feature vector
        return

    # returns vector of all state values
    def get_state_val(self):
        state_value = np.dot(self.all_state_features, self.weights)
        return state_value



#################### Running the experiment ###################################
def calc_RMSVE(state_values, true_state_values):
    msve = np.sum(np.square(true_state_values - state_values))
    rmsve = np.sqrt(msve)
    return rmsve

##################### Tunabled parameters #######################################
environment_parameters = {
    "num_states" : 500, 
    "start_state" : 250,
    "left_terminal_state" : 0,
    "right_terminal_state" : 501, 
    "seed" : 1,
    "discount_factor" : 1.0
}

agent_info = {"num_states": 500,
              "num_features": 10,
              "step_size": 0.01,
              "discount_factor": 0.9}

num_episodes = 2000
###############################################################################3
current_env = RandomWalkEnvironment()
current_env.env_init(environment_parameters)

test_agent = TDAgent()
test_agent.agent_init(agent_info)

episodic_rmsve = np.zeros(num_episodes)

for i in range(num_episodes): # episode
    is_terminal = False
    ctr = 0

    start_state = current_env.env_start()

    # first action
    current_action = test_agent.agent_start(start_state)

    while not is_terminal:
        reward_state_term = current_env.env_step(current_action)
        is_terminal = reward_state_term[2]
        ctr = ctr + 1
        # print("current_state:{0}".format(reward_state_term[1]))
        if not is_terminal:
            current_action = test_agent.agent_step(reward_state_term[0],reward_state_term[1])

    if is_terminal:
        test_agent.agent_end(reward_state_term[0])
        episodic_rmsve[i] = calc_RMSVE(test_agent.get_state_val(), current_env.true_state_values)

    # print("ctr={0}".format(ctr))


##################### ANALYSIS PART ############################
state_indices = np.linspace(current_env.left_terminal_state+1, current_env.right_terminal_state-1, num=current_env.num_states)
episode_indices = np.linspace(1, num_episodes, num=num_episodes)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(state_indices, test_agent.get_state_val())
ax1.set_xlabel('states')
ax1.set_ylabel('state values')

ax2.plot(episode_indices, episodic_rmsve)
ax2.set_xlabel('episode #')
ax2.set_ylabel('RMSVE w.r.t true state value')

plt.show(fig)
fig.savefig("TD_results.png")

