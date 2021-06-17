import random
import numpy
import copy
import math
from multi_armed_bandits import random_bandit, epsilon_greedy, boltzmann, UCB1

"""
 Base class of an autonomously acting and learning agent.
"""


class Agent:
    def __init__(self, params):
        self.nr_actions = params["nr_actions"]

    """
     Behavioral strategy of the agent. Maps states to actions.
    """

    def policy(self, state):
        pass

    """
     Learning method of the agent. Integrates experience into
     the agent's current knowledge.
    """

    def update(self, state, action, reward, next_state, done):
        pass


"""
 Randomly acting agent.
"""


class RandomAgent(Agent):
    def __init__(self, params):
        super(RandomAgent, self).__init__(params)

    def policy(self, state):
        return random.choice(range(self.nr_actions))


"""
 Autonomous agent using SARSA.
"""


class SARSALearner(Agent):
    def __init__(self, params):
        self.params = params
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.Q_values = {}
        self.alpha = params["alpha"]
        self.epsilon = params["epsilon"]

    def Q(self, state):
        state = numpy.array2string(state)
        if state not in self.Q_values:
            self.Q_values[state] = numpy.zeros(self.nr_actions)
        return self.Q_values[state]

    def policy(self, state):
        Q_values = self.Q(state)
        return epsilon_greedy(Q_values, None, epsilon=self.epsilon)

    def update(self, state, action, reward, next_state, done):
        future_action = self.policy(next_state)
        state = numpy.array2string(state)
        next_state = numpy.array2string(next_state)
        self.Q_values[state][action] = self.Q_values[state][action] + self.alpha * (
            reward
            + self.gamma * self.Q_values[next_state][future_action]
            - self.Q_values[state][action]
        )


"""
 Autonomous agent using SARSA.
"""


class QLearner(Agent):
    def __init__(self, params):
        self.params = params
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.Q_values = {}
        self.alpha = params["alpha"]
        self.epsilon = params["epsilon"]

    def Q(self, state):
        state = numpy.array2string(state)
        if state not in self.Q_values:
            self.Q_values[state] = numpy.zeros(self.nr_actions)
        return self.Q_values[state]

    def policy(self, state):
        Q_values = self.Q(state)
        return boltzmann(Q_values, None, 0.0001)
        # return epsilon_greedy(Q_values, None, epsilon=self.epsilon)

    def update(self, state, action, reward, next_state, done):
        state = numpy.array2string(state)
        _ = self.Q(next_state)
        next_state = numpy.array2string(next_state)
        future_action = numpy.argmax(self.Q_values[next_state])
        self.Q_values[state][action] = self.Q_values[state][action] + self.alpha * (
            reward
            + self.gamma * self.Q_values[next_state][future_action]
            - self.Q_values[state][action]
        )
