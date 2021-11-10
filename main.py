import rooms
import a2c
import matplotlib.pyplot as plot


def episode(env, agent, nr_episode=0):
    state = env.reset()
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    while not done:
        # env.render()
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, done)
        state = next_state
        discounted_return += reward * (discount_factor ** time_step)
        time_step += 1
    print(nr_episode, ":", discounted_return)
    return discounted_return


params = {}
training_episodes = 3000
max_steps = 100
# Domain setup
env = rooms.load_env("layouts/rooms_9_9_4.txt", "rooms.mp4", max_steps)
params["nr_actions"] = env.action_space.n
params["nr_input_features"] = env.observation_space.shape
params["env"] = env
# Hyperparameters
params["gamma"] = 0.99  # dinscount
params["alpha"] = 0.001  # lerning rate
print(f"params: {params}")

# Agent setup
agent = a2c.A2CLearner(params)
returns = [episode(env, agent, i) for i in range(training_episodes)]

x = range(training_episodes)
y = returns

plot.plot(x, y)
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("undiscounted return")
plot.show()

env.save_video()
