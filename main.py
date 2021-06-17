import rooms
import agent as a
import matplotlib.pyplot as plot


def episode(env, agent, nr_episode=0):
    # print(agent.epsilon)
    state = env.reset()
    print(state)
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    linear_decay = 0.0003
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, done)
        state = next_state
        discounted_return += reward * (discount_factor ** time_step)
        time_step += 1
    if nr_episode > 0.3 * training_episodes and agent.epsilon != 0:
        if agent.epsilon > linear_decay:
            agent.epsilon -= linear_decay
        else:
            agent.epsilon = 0

    print(nr_episode, ":", discounted_return)
    return discounted_return


params = {}
training_episodes = 500
env = rooms.load_env("layouts/rooms_9_9_4.txt", "rooms_9_500_100.mp4", 100)
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["horizon"] = 10
params["simulations"] = 100
params["alpha"] = 0.2
params["epsilon"] = 0.1
params["env"] = env

# agent = a.RandomAgent(params)
agent = a.SARSALearner(params)
# agent = a.QLearner(params)

returns = [episode(env, agent, i) for i in range(training_episodes)]

x = range(training_episodes)
y = returns

plot.plot(x, y)
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("discounted return")
plot.show()

env.save_video()
