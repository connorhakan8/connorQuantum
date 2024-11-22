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
    def __init__(self):
        print("This is a placeholder method.")

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

    def _step(self):
        print("This is a placeholder method that might go here later.")


