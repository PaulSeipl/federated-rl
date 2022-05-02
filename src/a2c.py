import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from copy import copy, deepcopy


class A2CNet(nn.Module):
    def __init__(self, nr_input_features, nr_actions):
        super(A2CNet, self).__init__()
        nr_hidden_units = 64
        self.fc_net = nn.Sequential(
            nn.Linear(nr_input_features, nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
        )
        self.nr_input_features = nr_input_features
        self.action_head = nn.Linear(nr_hidden_units, nr_actions)
        self.value_head = nn.Linear(nr_hidden_units, 1)

    def forward(self, x):
        x = x.view(x.size(0), self.nr_input_features)
        x = self.fc_net(x)
        return F.softmax(self.action_head(x), dim=-1), self.value_head(x)


"""
 Autonomous agent using Synchronous Actor-Critic.
"""


class A2CLearner:
    def __init__(self, params):
        self.eps = numpy.finfo(numpy.float32).eps.item()
        self.name = params["name"]
        self.gamma = params["gamma"]
        self.nr_actions = params["nr_actions"]
        self.alpha = params["alpha"]
        self.nr_input_features = numpy.prod(params["nr_input_features"])
        self.max_done_counter = params["max_done_counter"]
        self.done_counter = 0
        self.transitions = []
        self.batch = []
        self.device = torch.device("cpu")
        self.a2c_net = A2CNet(self.nr_input_features, self.nr_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.a2c_net.parameters(), lr=params["alpha"])

    def get_parameters(self):
        return self.a2c_net.parameters()

    def get_state_dict(self):
        return self.a2c_net.state_dict()

    def get_state_dict_copy(self):
        return deepcopy(self.a2c_net.state_dict())

    def save_state_dict(self):
        torch.save(self.a2c_net.state_dict(), f"../models/{self.name}_dict.pt")

    def load_state_dict(self, state_dict):
        self.a2c_net.load_state_dict(state_dict)

    def load_state_dict_file(self):
        self.a2c_net.load_state_dict(torch.load(f"../models/{self.name}_dict.pt"))

    def save_net(self, model_name=None):
        torch.save(self.a2c_net, f"../models/{self.name}_model.pt") if not (
            model_name
        ) else torch.save(self.a2c_net, f"./models/{model_name}_model.pt")

    def load_net_file(self, model_name=None):
        self.a2c_net = (
            torch.load(f"../models/{model_name}_model.pt")
            if model_name
            else torch.load(f"../models/{self.name}_model.pt")
        )

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name

    """
     Gets argmax of actions properties
    """

    def policy_deterministic(self, state):
        action_probs, _ = self.predict_policy([state])
        action = torch.argmax(action_probs)
        return action.item()

    """
     Samples a new action using the policy network.
    """

    def policy(self, state):
        action_probs, _ = self.predict_policy([state])
        m = Categorical(action_probs)
        action = m.sample()
        return action.item()

    """
     Predicts the action probabilities.
    """

    def predict_policy(self, states):
        states = torch.tensor(states, device=self.device, dtype=torch.float)
        return self.a2c_net(states)

    """
     Performs a learning update of the currently learned policy and value function.
    """

    def update(self, state, action, reward, next_state, done):
        self.transitions.append((state, action, reward, next_state, done))
        loss = None

        if done:
            self.done_counter += 1
            self.batch.append(deepcopy(self.transitions))
            self.transitions.clear()

        if self.done_counter == self.max_done_counter:
            self.done_counter = 0

            for transitions in self.batch:
                states, actions, rewards, next_states, dones = tuple(zip(*transitions))
                discounted_returns = []

                # Calculate and normalize discounted returns
                R = 0
                for reward in reversed(rewards):
                    R = reward + self.gamma * R
                    discounted_returns.append(R)
                discounted_returns.reverse()
                states = states
                next_states = next_states
                rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
                discounted_returns = torch.tensor(
                    discounted_returns, device=self.device, dtype=torch.float
                ).detach()
                normalized_returns = discounted_returns - discounted_returns.mean()
                normalized_returns /= discounted_returns.std() + self.eps

                # Calculate losses of policy and value function
                actions = torch.tensor(actions, device=self.device, dtype=torch.long)
                action_probs, state_values = self.predict_policy(states)
                states = torch.tensor(states, device=self.device, dtype=torch.float)
                policy_losses = []
                value_losses = []
                for probs, action, value, R in zip(
                    action_probs, actions, state_values, normalized_returns
                ):
                    advantage = R - value.item()
                    m = Categorical(probs)
                    policy_losses.append(-m.log_prob(action) * advantage)
                    value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
                loss = (
                    torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
                )

                # Optimize joint loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            self.batch.clear()
