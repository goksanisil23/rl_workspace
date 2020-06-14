# libraries for tiling
import numpy as np
import matplotlib.pyplot as plt
import tiles3 as tc

# libraries for environment
import cv2
from InvertedPendulum import InvertedPendulum
from scipy.integrate import solve_ivp
import random
from tqdm import tqdm

# libraries for DNN
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import rmsprop, Adam
from collections import deque
from statistics import mean
import h5py

## DNN hyperparameters
LEARNING_RATE = 1e-3
MAX_MEMORY = 1000000
BATCH_SIZE = 20
GAMMA = 0.95
EXPLORATION_DECAY = 0.995
EXPLORATION_MIN = 0.01



#################### Inverted Pendulum Environment implementation ########################### 
g = 9.8 # gravitational acc.
L = 1.5 # length of pendulum
m = 1.0 # mass of bob (kg.)
M = 5.0 # mass of cart(kg.)

d1 = 0.5 # damping coefficient for cart
d2 = 1 # damping coefficient for pendulum

action_duration = 0.2 # when an action is applied, how long should it last
dynamics_resolution = 0.1 # determines how many solutions to ODE
sol_step_no = int(action_duration/dynamics_resolution)    

applied_force = -100

inv_pend_renderer = InvertedPendulum()

def pendulum_dynamics(t,states):

    x_ddot = applied_force - m*L*states[3]*states[3] * np.cos( states[2] ) + m*g*np.cos(states[2]) *  np.sin(states[2])
    x_ddot = x_ddot / ( M+m-m* np.sin(states[2])* np.sin(states[2]) )

    theta_ddot = -g/L * np.cos( states[2] ) - 1./L * np.sin( states[2] ) * x_ddot

    damping_theta =  - d2*states[3]
    damping_x =  - d1*states[1]

    return [ states[1], x_ddot + damping_x, states[3], theta_ddot + damping_theta ]  

class InvertedPendulum():
    def env_init(self, env_info={}):

        # states : [ x, x_dot, theta, theta_dot]
        self.init_state = env_info["start_state"]

    def env_start(self):
        
        reward = 0.0
        self.states = self.init_state
        self.is_terminal = False

        self.reward_state_term = (reward, self.states, self.is_terminal)

        return self.reward_state_term[1] # returning first state from the env   

    def env_step(self, action, ctr, animation_on):
        # states : [ x, x_dot, theta, theta_dot]

        global applied_force

        if action==0:
            applied_force = 55 #95
        elif action==1:
            applied_force = -55 #-95
        else:
            applied_force = 0

        t_start = ctr*action_duration
        t_end = ((ctr+1)*action_duration)-dynamics_resolution
        t_interval = np.linspace(t_start, t_end, sol_step_no)
        sol = solve_ivp(pendulum_dynamics, [t_start, t_end], self.states, t_eval= t_interval)
        self.states = [sol.y[0,-1], sol.y[1,-1], sol.y[2,-1], sol.y[3,-1] ] # take the last state from the solution


        ######## for rendering ############
        if animation_on:
            global inv_pend_renderer
            for ii, tt in enumerate(sol.t):
                rendered = inv_pend_renderer.step( [sol.y[0,ii], sol.y[1,ii], sol.y[2,ii], sol.y[3,ii] ], tt )
                cv2.imshow( 'im', rendered )
                cv2.moveWindow( 'im', 100, 100 )

                if cv2.waitKey(30) == ord('q'):
                    break       
        ##################################### 
        reward = 0
        if abs(self.states[0])>10: # should not exceed environment boundary
            # reward = -1.0
            self.is_terminal = True
            # print("out of bounds")
        elif (abs( (self.states[2]%(2*np.pi)) - (np.pi/2) ) > np.pi/180*20): # 20 degrees left/right
            # reward = -1.0
            self.is_terminal = True
            # print("out of operation range")
        else:
            # out_of_bounds = False
            # action_exists = False
            # if abs(self.states[0])>10 :
            #     out_of_bounds = True
            
            # if abs(applied_force) > 0:
            #     action_exists = True 
            # reward = 1
            
            reward = 1 -abs( (self.states[2]%(2*np.pi)) - (np.pi/2) ) / (2*np.pi) #normalized to [0,1]
            # print("reward:", str(reward))
            # reward = 1*np.sin(self.states[2]) - 0.2*abs(self.states[3]) #- 5*out_of_bounds+
            # reward = -10*(abs( (self.states[2]%(2*np.pi)) - (np.pi/2) ) ) - 1*abs(self.states[3]) 
            # reward = 10*(abs( (self.states[2]%(2*np.pi)) - (np.pi/2) ) < np.pi/10)- 10*out_of_bounds

            # if (abs( (self.states[2]%(2*np.pi)) - (np.pi/2) ) < np.pi/400) & (abs(self.states[3])<0.01):
            #     reward = reward + 800 # extra bonus
            #     print("extra bonus 1")
            # elif (abs( (self.states[2]%(2*np.pi)) - (np.pi/2) ) < np.pi/200) & (abs(self.states[3])<0.05):
            #     reward = reward + 400 # extra bonus
            #     print("extra bonus 2")
            # elif (abs( (self.states[2]%(2*np.pi)) - (np.pi/2) ) < np.pi/10) & (abs(self.states[3])<0.5):
            #     reward = reward + 100 # extra bonus
            #     print("extra bonus 3")                         

        # self.reward_state_term = (reward, self.states, self.is_terminal)
        

        return (reward, self.states, self.is_terminal)


##################### Agent #########################################################
#  Helper function that Takes in a list of q_values and returns the index of the item with the highest value. Breaks ties randomly.
def argmax(q_values):
    top = float("-inf")
    ties = []
    for i in range(len(q_values)):
        if q_values[i] > top:
            top, ties = q_values[i], [i]
        elif q_values[i] == top:
            ties.append(i)
    
    ind = np.random.choice(ties)
    return ind  

class SarsaAgent():
    def __init__(self):
        self.load_model = False
        self.action_space = [0,1,2]
        self.action_size = len(self.action_space)
        self.state_size = 4  
        self.discount_factor = 0.99    
        self.learning_rate = 0.001         

        self.epsilon = 1.  # exploration
        self.epsilon_decay = .99995
        self.epsilon_min = 0.01
        self.model = self.build_model()

        if self.load_model:
            self.epsilon = 0.05
            self.model.load_weights('deep_sarsa_trained.h5')

    # approximate Q function using NN (input: states of pendulum -> output: Q value of each action)
    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model  

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)
        else:
            # Predict the reward value based on the given state
            state = np.float32(state)
            q_values = self.model.predict(state)
            return argmax(q_values[0])

    def get_demo_action(self, state):
        state = np.float32(state)
        q_values = self.model.predict(state)
        return argmax(q_values[0])            


    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            print("epsilon: ", str(self.epsilon))

        state = np.float32(state)
        next_state = np.float32(next_state)
        target = self.model.predict(state)[0]
        # get maximum Q value at s' (next state) from target model (already decided next state and action --> sampled from policy)
        if done:
            target[action] = reward
        else:
            target[action] = (reward + self.discount_factor * self.model.predict(next_state)[0][next_action])

        target = np.reshape(target, [1, self.action_size])
        # make minibatch which includes target q value and predicted q value
        # and do the model fit! (as we are trying to minimize the prediction that is bootstrapped from the sample from the policy)
        self.model.fit(state, target, epochs=1, verbose=0)   # bootstrapped estimation = target = ground truth for this DNN           

#################### Running the experiment ###################################
current_env = InvertedPendulum()

env_info = {"start_state": [ -1.0, 0., np.pi/2 - np.pi/50  , 0. ]}
current_env.env_init(env_info)

test_agent = SarsaAgent()

num_episodes = 1500
num_iterations = 500

scores, episodes = [], []
###############################################################################3
animation_on = False
for kk in tqdm(range(num_episodes)): # episode
    done = False
    score = 0

    ctr = 0

    state = current_env.env_start()
    state = np.reshape(state, [1, test_agent.state_size])

    while ((not current_env.is_terminal) and (ctr < num_iterations)):
        # get action for the current state and go one step in environment
        current_action = test_agent.get_action(state)
        reward, next_state, is_terminal = current_env.env_step(current_action, ctr, animation_on)
        next_state = np.reshape(next_state, [1, test_agent.state_size])
        next_action = test_agent.get_action(next_state)
        ctr = ctr + 1
        if(ctr==num_iterations):
            is_terminal = True
        test_agent.train_model(state, current_action, reward, next_state, next_action, is_terminal)
        state = next_state
        # state = copy.deepcopy(next_state)
        
        score += reward

        if(ctr>=num_iterations):
            print("max iterations reached")
        
        if is_terminal:
            scores.append(score)
            episodes.append(kk)

    # print(ctr)



############### Test the learnt agent ###############################
plt.plot(episodes, scores, label="score per run")
plt.show()
animation_on = True
input("start the learnt agent")
# while True:
for kk in range(1000):
    ctr = 0
    state = current_env.env_start()
    state = np.reshape(state, [1, test_agent.state_size])

    while(not current_env.is_terminal):
        current_action = test_agent.get_demo_action(state)
        reward, next_state, is_terminal = current_env.env_step(current_action, ctr, animation_on)
        next_state = np.reshape(next_state, [1, test_agent.state_size])
        state = next_state
        ctr = ctr + 1

        
