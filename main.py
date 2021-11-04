import rooms

# import agent as a
import matplotlib.pyplot as plot
import numpy

from a2c import A2CLearner


def flatNumpyState(state):
    return numpy.array(state).flatten()


def episode(env, agent, nr_episode=0):
    # print(agent.epsilon)
    rawState = env.reset()
    flatState = flatNumpyState(rawState)
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(flatState)
        # 2. Execute selected action
        nextRawState, reward, done, _ = env.step(action)
        nextFlatState = flatNumpyState(nextRawState)
        # 3. Integrate new experience into agent
        agent.update(flatState, action, reward, nextFlatState, done)
        flatState = flatNumpyState(nextFlatState)
        discounted_return += reward * (discount_factor ** time_step)
        time_step += 1

    print(nr_episode, ":", discounted_return)
    return discounted_return


params = {}
training_episodes = 5000
max_steps = 100
env = rooms.load_env("layouts/rooms_9_9_4.txt", "rooms_9_500_100.mp4", max_steps)
initialState = env.reset()
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["alpha"] = 0.2  # learning rate
params["env"] = env
params["nr_input_features"] = len(flatNumpyState(initialState))

a2cAgent = A2CLearner(params)

returns = [episode(env, a2cAgent, i) for i in range(training_episodes)]

x = range(training_episodes)
y = returns

plot.plot(x, y)
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("discounted return")
plot.show()

env.save_video()
