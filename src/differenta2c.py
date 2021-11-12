import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt
import rooms

# helper function to convert numpy arrays to tensors
def t(x):
    return torch.from_numpy(x).float()


# Actor module, categorical actions only
class Actor(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, n_actions),
            nn.Softmax(),
        )

    def forward(self, X):
        return self.model(X)


# Critic module
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, X):
        return self.model(X)


# env = gym.make("CartPole-v1")

# result = entry_point.load(False)


def flatNumpyState(state):
    return np.array(state).flatten()


params = {}
training_episodes = 5000
max_steps = 100
env = rooms.load_env("layouts/rooms_9_9_4.txt", "rooms_9_500_100.mp4", max_steps)
initialState = env.reset()
print(env.action_space.n)
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["alpha"] = 0.2  # learning rate
params["env"] = env
params["nr_input_features"] = len(flatNumpyState(initialState))
# config
state_dim = len(flatNumpyState(initialState))
n_actions = env.action_space.n
actor = Actor(state_dim, n_actions)
critic = Critic(state_dim)
adam_actor = torch.optim.Adam(actor.parameters(), lr=1e-3)
adam_critic = torch.optim.Adam(critic.parameters(), lr=1e-3)
gamma = 0.99

episode_rewards = []

for i in range(1000):
    done = False
    total_reward = 0
    rawState = env.reset()
    state = flatNumpyState(rawState)

    while not done:
        probs = actor(t(state))
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()

        raw_next_state, reward, done, info = env.step(action.detach().data.numpy())
        next_state = flatNumpyState(raw_next_state)
        advantage = (
            reward + (1 - done) * gamma * critic(t(next_state)) - critic(t(state))
        )

        total_reward += reward
        state = next_state

        critic_loss = advantage.pow(2).mean()
        adam_critic.zero_grad()
        critic_loss.backward()
        adam_critic.step()

        actor_loss = -dist.log_prob(action) * advantage.detach()
        adam_actor.zero_grad()
        actor_loss.backward()
        adam_actor.step()

    episode_rewards.append(total_reward)

plt.scatter(np.arange(len(episode_rewards)), episode_rewards, s=2)
plt.title("Total reward per episode (online)")
plt.ylabel("reward")
plt.xlabel("episode")
plt.show()
env.save_video()
