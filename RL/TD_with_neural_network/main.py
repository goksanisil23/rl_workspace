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

######################### Some Helper Funnctions for NN #############################
# Since the input of the NN is a sparse vector (all zeros except 1 index), we can use a more efficient matrix multiplication
def my_matmul(x1, x2):
    result = np.zeros((x1.shape[0], x2.shape[1]))
    x1_non_zero_indices = x1.nonzero()
    if x1.shape[0] == 1 and len(x1_non_zero_indices[1]) == 1:
        result = x2[x1_non_zero_indices[1], :]
    elif x1.shape[1] == 1 and len(x1_non_zero_indices[0]) == 1:
        result[x1_non_zero_indices[0], :] = x2 * x1[x1_non_zero_indices[0], 0]
    else:
        result = np.matmul(x1, x2)
    return result

# Compute the value of input state s, given the weights of a NN
def get_value(s, weights):
    psi = my_matmul(s, weights[0]["W"]) + weights[0]["b"]
    x = np.maximum(psi, 0)
    v = my_matmul(x, weights[1]["W"]) + weights[1]["b"]
    
    return v

# computes gradient of value function for a given input via backpropogation
# example weights: weights[0]["W"],weights[0]["b"],weights[1]["W"],weights[1]["b"]
def get_gradient(s,weights):
    grads = [dict() for i in range(len(weights))]

    x = np.maximum(my_matmul(s, weights[0]["W"]) + weights[0]["b"], 0)

    grads[0]["W"] = my_matmul(s.T, (weights[1]["W"].T * (x>0))) 
    grads[0]["b"] = weights[1]["W"].T * (x > 0)
    grads[1]["W"] = x.T
    grads[1]["b"] = 1

    return grads

def one_hot(state, num_states):
    one_hot_vector = np.zeros((1, num_states))
    one_hot_vector[0,int((state-1))] = 1

    return one_hot_vector

################ Stochastic Gradient Descent Implementation ######################
class SGD():
    def __init__(self):
        pass

    def optimizer_init(self, optimizer_info):
        self.step_size = optimizer_info.get("step_size")

    # g is the direction term: g = (TD-error)*delta(value_func)
    def update_weights(self, weights, g):
        for i in range(len(weights)):
            for param in weights[i].keys():
                weights[i][param] += self.step_size * g[i][param]

        return weights

class Adam():
    def __init__(self):
        pass
    
    def optimizer_init(self, optimizer_info):
        #parameters needed for Adam algorithm
        self.num_states = optimizer_info.get("num_states")
        self.num_hidden_layer = optimizer_info.get("num_hidden_layer")
        self.num_hidden_units = optimizer_info.get("num_hidden_units")

        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")
        self.beta_v = optimizer_info.get("beta_v")
        self.epsilon = optimizer_info.get("epsilon")

        self.layer_size = np.array([self.num_states, self.num_hidden_units, 1])

        # initialize Adam's m and v
        self.m = [dict() for i in range(self.num_hidden_layer+1)]
        self.v = [dict() for i in range(self.num_hidden_layer+1)]

        for i in range(self.num_hidden_layer+1):
            self.m[i]["W"] = np.zeros((self.layer_size[i], self.layer_size[i+1]))
            self.m[i]["b"] = np.zeros((1, self.layer_size[i+1]))
            self.v[i]["W"] = np.zeros((self.layer_size[i], self.layer_size[i+1]))
            self.v[i]["b"] = np.zeros((1, self.layer_size[i+1]))

            self.beta_m_product = self.beta_m
            self.beta_v_product = self.beta_v

    def update_weights(self, weights, g):
        for i in range(len(weights)):
            for param in weights[i].keys():
                self.m[i][param] = self.beta_m * self.m[i][param] + (1 - self.beta_m) * g[i][param]
                self.v[i][param] = self.beta_v * self.v[i][param] + (1 - self.beta_v) * (g[i][param] * g[i][param])

                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)

                weights[i][param] += self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v

        return weights

############################## Agent Implementation ##########################3
class TDAgent():
    def __init__(self):
        self.name = "td_agent"
        pass

    def agent_init(self, agent_info={}):

        # Set random seed for weights initialization for each run
        self.rand_generator = np.random.RandomState(agent_info.get("seed")) 
        
        # Set random seed for policy for each run
        self.policy_rand_generator = np.random.RandomState(agent_info.get("seed"))

        self.num_states = agent_info.get("num_states")
        self.num_hidden_layer = agent_info.get("num_hidden_layer")
        self.num_hidden_units = agent_info.get("num_hidden_units")
        self.discount_factor = agent_info.get("discount_factor")

        self.layer_size = np.array([self.num_states, self.num_hidden_units, 1])

        self.weights = [dict() for i in range(self.num_hidden_layer+1)]

        for i in range(self.num_hidden_layer+1):
            ins, outs = self.layer_size[i], self.layer_size[i+1]
            self.weights[i]['W'] = self.rand_generator.normal(0, np.sqrt(2/ins), (ins, outs))
            self.weights[i]['b'] = self.rand_generator.normal(0, np.sqrt(2/ins), (1, outs))
        
        self.optimizer = Adam()
        optimizer_info = {"num_states": agent_info["num_states"], "num_hidden_layer": agent_info["num_hidden_layer"],
                          "num_hidden_units": agent_info["num_hidden_units"],"step_size": agent_info["step_size"],
                          "beta_m": agent_info["beta_m"], "beta_v": agent_info["beta_v"], "epsilon": agent_info["epsilon"]}
        self.optimizer.optimizer_init(optimizer_info)

        self.last_state = None
        self.last_action = None

    def agent_policy(self, state):
        chosen_action = self.policy_rand_generator.choice([0,1])
        return chosen_action

    def agent_start(self, state):
        self.last_state = state
        self.last_action = self.agent_policy(state)
        
        return self.last_action
    
    def agent_step(self, reward, state):
        last_state_vec = one_hot(self.last_state, self.num_states)
        last_value = get_value(last_state_vec, self.weights)

        state_vec = one_hot(state, self.num_states)
        value = get_value(state_vec, self.weights)

        delta = reward + self.discount_factor * value - last_value

        grads = get_gradient(last_state_vec, self.weights)

        g = [dict() for i in range(self.num_hidden_layer+1)]

        for i in range(self.num_hidden_layer+1):
            for param in self.weights[i].keys():
                g[i][param] = delta * grads[i][param]

        self.weights = self.optimizer.update_weights(self.weights, g)

        self.last_state = state
        self.last_action = self.agent_policy(state)

        return self.last_action

    def agent_end(self, reward):

        last_state_vec = one_hot(self.last_state, self.num_states)
        last_value = get_value(last_state_vec, self.weights)
        
        delta = reward - last_value

        grads = get_gradient(last_state_vec, self.weights)

        g = [dict() for i in range(self.num_hidden_layer+1)]
        for i in range(self.num_hidden_layer+1):
            for param in self.weights[i].keys():
                g[i][param] = delta * grads[i][param]

        self.weights = self.optimizer.update_weights(self.weights, g)

    def get_state_value(self):
        state_value = np.zeros(self.num_states)
        for state in range(1, self.num_states + 1):
            s = one_hot(state, self.num_states)
            state_value[state - 1] = get_value(s, self.weights)
        
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
    "discount_factor" : 1.0
}

# Agent parameters
agent_parameters = {
    "num_hidden_layer": 1,
    "num_hidden_units": 100,
    "step_size": 0.001,
    "beta_m": 0.9,
    "beta_v": 0.999,
    "epsilon": 0.0001,
}

agent_info = {"num_states": environment_parameters["num_states"],
                "num_hidden_layer": agent_parameters["num_hidden_layer"],
                "num_hidden_units": agent_parameters["num_hidden_units"],
                "step_size": agent_parameters["step_size"],
                "discount_factor": environment_parameters["discount_factor"],
                "beta_m": agent_parameters["beta_m"],
                "beta_v": agent_parameters["beta_v"],
                "epsilon": agent_parameters["epsilon"]
                }

num_episodes = 2000
###############################################################################
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
        if not is_terminal:
            current_action = test_agent.agent_step(reward_state_term[0],reward_state_term[1])

    if is_terminal:
        test_agent.agent_end(reward_state_term[0])
        episodic_rmsve[i] = calc_RMSVE(test_agent.get_state_value(), current_env.true_state_values)

    # print("ctr={0}".format(ctr))

##################### ANALYSIS PART ############################
state_indices = np.linspace(current_env.left_terminal_state+1, current_env.right_terminal_state-1, num=current_env.num_states)
episode_indices = np.linspace(1, num_episodes, num=num_episodes)

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

ax1.plot(state_indices, test_agent.get_state_value())
ax1.set_xlabel('states')
ax1.set_ylabel('state values')

ax2.plot(episode_indices, episodic_rmsve)
ax2.set_xlabel('episode #')
ax2.set_ylabel('RMSVE w.r.t true state value')

plt.show(fig)
fig.savefig("TD_NN_results.png")