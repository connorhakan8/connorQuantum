#this ia a mix of me trying to figure stuff out generally on my own
#i sometimes go back to the original file for inspiration on how to approach something
#other times i just write pseudocode-like code (should look like python code, but might not run)


import gym
from gym import spaces
import hashlib
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.circuit.library import HGate, CXGate, SGate, TGate, XGate, YGate, ZGate
from qiskit.quantum_info import Operator

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

CNOT = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
])

bell_state_unitary = Operator(CNOT) @ Operator(np.kron(H, np.eye(2))) 

cz_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
])

swap_matrix = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1]
])

iswap_matrix = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1]
])

class QuantumEnv(gym.Env):
    def __init__(self):
        self.num_qubits = 2
        self.circuit = QuantumCircuit(self.num_qubits)
        self.space = 0
        self.model = 0
        self.discount_rate = 0.9
        self.learning_rate = 0.8

    def get_state(self):
        #This is confusing me. How should I represent the state?
        #The original program used some kind of index. How is that used?
        
    
class QLearningAgent:
    def __init__(self, learning_rate, discount_factor, exploration_prob, state_size, action_size):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.state_size = state_size
        self.action_size = action_size
        #we make a table filled with zeros that has dimensions state_size x action_size
        self.q_table = np.zeros((state_size, action_size))

    #all the possible moves. We add a specific gate (h gate, t gate, etc)
    def _move(self, action):
        moves = [
            [HGate, [0]]
            [HGate, [1]]
            [CXGate, [0, 1]]
            [CXGate, [1, 0]]
            [SGate, [0]]
            [SGate, [1]]
            [TGate, [0]]
            [TGate, [1]]
            [XGate, [0]]
            [XGate, [1]]
            [YGate, [0]]
            [YGate, [1]]
            [ZGate, [0]]
            [ZGate, [1]]

        ]

        if np.random.rand() < self.exploration_prob:
            action = np.random.randint(0, self.state_size)
        else:
            action = np.argmax(self.q_table[current_action])
        

    def _step(self):
        print("Placeholder")

    #rudimentary reward function
    def _reward(self):
        if(current_state == target_state):
            reward = 10
        else:
            reward = -1

    def _update_q_table(): #formula from geeksforgeeks, somewhat adopted for this program
        self.q_table[self.current_state, self.action] += self.learning_rate * (self.reward + self.discount_factor * np.max(self.q_table[next_state]) - self.q_table[self.current_state, self.action])

            
    #current status: getting a good idea on how reinforcement learning works,
    #having trouble implementing this for a quantum system.
