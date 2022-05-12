import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy as sp
import networkx as nx

class SingleActorCritic():
    
    def __init__(self, state_size, action_size, alpha, gamma, beta):
        
        self.state_size = state_size
        self.action_size = action_size
        self.sv = np.zeros(state_size)
        
        self.gamma = gamma
        self.beta = beta
        self.alpha_a = alpha[0]
        self.alpha_c = alpha[1]
        
        self.V = np.zeros([state_size]) # state value function
        self.tA = np.random.uniform(0,0.01, [state_size, action_size]) # advantage 

    def sample_action(self, tA, beta):
        # softmax
        x = tA - tA.max(axis = None, keepdims = True)
        y = np.exp(x * self.beta)
        action_prob = y / y.sum(axis = None, keepdims = True)
        
        return np.argmax(np.random.multinomial(1, action_prob, size=1))
    
    # make this plot a bit prettier
    def plot_transfer(self):
        x = np.linspace(-10,10,1000)
        plt.plot(x,self.nF(x,1))
        plt.plot(x,self.nF(x,-1))
        plt.show()
    
    def update_td(self, current_exp):
        
        s = current_exp[0] # current state
        a = current_exp[1] # current action
        s1 = current_exp[2] # next state
        a1 = current_exp[3] # next action
        r = current_exp[4] # reward
        t_flag = current_exp[5] # terminal flag
        
        # update td-error
        
        if t_flag == False:
            delta = r + self.gamma * self.V[s1] - self.V[s]
        else:
            delta = r - self.V[s]
        
        # update critic
        self.V[s] += self.alpha_c * delta
        
        # update actors
        self.tA[s,a] += self.alpha_a * delta  # direct pathway actor
        
        return delta
    
    def get_V(self):
        
        return np.reshape(self.V, [grid_size, grid_size])
    
    
    
class SARSA_agent():
    
    def __init__(self, state_size, action_size, alpha, gamma, beta):
        
        self.state_size = state_size
        self.action_size = action_size
        
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        
        # state-action value function
        self.tA = np.random.uniform(0,0.01, [state_size, action_size]) # action preference

    def sample_action(self, tA, beta):
        
        # softmax
        x = tA - tA.max(axis=None, keepdims=True)
        y = np.exp(x * self.beta)
        action_prob = y / y.sum(axis=None, keepdims=True)
        
        return np.argmax(np.random.multinomial(1, action_prob, size=1))
    
    def update_td(self, current_exp):
        
        s = current_exp[0] # current state
        a = current_exp[1] # current action
        s1 = current_exp[2] # next state
        a1 = current_exp[3] # next action
        r = current_exp[4] # reward
        t_flag = current_exp[5] # terminal flag
        
        if t_flag:
            delta = r - self.tA[s,a]
        else:
            a1 = np.argmax(self.tA[s1,:])
            delta = r + self.gamma * self.tA[s1,a1] - self.tA[s,a]
            
        # update action-values
        self.tA[s,a] += self.alpha * delta
        
        return delta
        
    def get_V(self):
        
        v = np.zeros([self.state_size])
        for i in range(self.state_size):
            v[i] =  np.mean(self.tA[i,:])
        
        return np.reshape(v, [grid_size, grid_size])