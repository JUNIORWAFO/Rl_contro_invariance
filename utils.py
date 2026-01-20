import control as ct
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull,QhullError
import gym
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from collections import deque

def generate_model_spring_mass_dampener(n, k, c, M, dt=0.1):
    M = np.diag(1/np.array(M))
    A = np.zeros((2*n, 2*n))
    B = np.zeros((2*n, n))
    A[0:n,n:] = np.eye(n)
    B[0:n,:] = M
    K = np.zeros((n,n))
    C = np.zeros((n,n))
    for i in range(n):
        succ = (i+1) % n
        prec = (i-1) % n
        h = int(i<n-1)
        C[i,i] = c[i]+ c[succ]* h
        K[i,i] = k[i]+ k[succ]* h
        K[i,succ] = -k[succ]  * h
        K[i,prec] = -k[i] * int(i>0) + K[i,succ] * int(prec == succ)
        C[i,succ] = -c[succ]  * h
        C[i,prec] = -c[i] * int(i>0) +C[i,succ] * int(prec == succ)
    A[n:,0:n] = - M @ K
    A[n:,n:] =  M @ C
    sys = ct.ss(A, B, np.eye(2*n), np.zeros((2*n, n)))
    sys_d = ct.c2d(sys, dt)
    # A = np.eye(2*n) + dt * A
    # B = dt *B
    return sys_d.A,sys_d.B

class RNNActor(nn.Module):
    def __init__(self, input_size, hidden_size, action_size, num_layers=1):
        super(RNNActor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, action_size)

    def forward(self, state, hidden):
        # Assuming state is of shape (batch_size, seq_len, input_size)
        out, hidden = self.rnn(state, hidden)
        try:
            out = self.fc(out[-1, : , :]) # Take only the last output
        except:
            out = self.fc(out[-1, :]) # Take only the last output
        out = torch.sigmoid(out) # Ensure output is bounded between  and 1
        return out, hidden

    def init_hidden(self, batch_size):
         return torch.zeros(self.num_layers, batch_size, self.hidden_size)

# class RNNActor(nn.Module):
#     def __init__(self, input_size, hidden_size, action_size, num_layers=1):
#         super(RNNActor, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, action_size)
#         self.initial_hidden = nn.Parameter(torch.randn(num_layers, 1, hidden_size)) # Define initial hidden state as a parameter

#     def forward(self, state, hidden):
#         # Assuming state is of shape (batch_size, seq_len, input_size)
#         out, hidden = self.rnn(state, hidden)
#         out = self.fc(out[:, -1, :]) # Take only the last output
#         out = torch.tanh(out) # Ensure output is bounded between -1 and 1
#         return out, hidden

#     def init_hidden(self, batch_size):
#         return self.initial_hidden.repeat(1, batch_size, 1)
#critic network
class Critic(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size + hidden_size + action_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state, action, hidden):
        x = torch.cat((state, action , hidden), dim=2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
# Define the DQN Network
class DQN(nn.Module):
    def __init__(self, input_size, output_size, last_hidden =None):
        super(DQN, self).__init__()
        if last_hidden is None:
            self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.SELU(),
            nn.Linear(16, 32),
            nn.SELU(),
            nn.Linear(32, 64),
            nn.SELU(),
            nn.Linear(64, 32),
            nn.SELU(),
            nn.Linear(32, 16),
            nn.SELU(),
            nn.Linear(16, output_size))
        else:
            self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_size),
            nn.Hardtanh())
    def forward(self, x):
        return self.model(x)

# Replay Memory
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        self.memory.append((state, action , next_state, reward, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)