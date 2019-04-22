import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6, alpha=1e-2, eps_start=1.0, eps_decay=.99999, eps_min=1e20, algorithm="expected-sarsa"):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.alpha = alpha


    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if random.random() > self.eps: # select greedy action with probability epsilon
            return np.argmax(self.Q[state])
        else:                     # otherwise, select an action randomly
            return random.choice(np.arange(self.nA))

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        Q = self.Q
        nA = self.nA
        eps = self.eps
        alpha = self.alpha

        new_value = 0
        current = Q[state][action]                                  # estimate in Q-table (for current state, action pair)
        
        algorithm = "q-learning"

        if (algorithm == "q-learning"):
            Qsa_next = np.max(Q[next_state]) if next_state is not None else 0  # value of next state 
            target = reward + Qsa_next               # construct TD target
            new_value = current + (alpha * (target - current)) # get updated value 

        if (algorithm == "expected-sarsa"):
            policy_s = np.ones(nA) * eps / nA                           # current policy (for next state S')
            policy_s[np.argmax(Q[next_state])] = 1 - eps + (eps / nA)   # greedy action
            Qsa_next = np.dot(Q[next_state], policy_s)                  # get value of state at next time step
            target = reward + Qsa_next                                  # construct target
            new_value = current + (alpha * (target - current))          # get updated value 

        self.Q[state][action] = new_value

        self.eps = max(self.eps*self.eps_decay, self.eps_min)
        
