from src import rooms, a2c
import matplotlib.pyplot as plot
from os import walk


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


def create_plot(title, x_data, x_label, y_data, y_label):
    plot.plot(x_data, y_data)
    plot.title(title)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.show()


def get_room_files(rooms_dir):
    (_, _, room_files) = next(walk(f"./layouts/{rooms_dir}"), (None, None, []))
    return room_files


def get_room_path(rooms_dir, room_file):
    return f"layouts/{rooms_dir}/{room_file}"


def get_movie_file_path(room_file, max_steps):
    return f"movies/{room_file.replace('.txt', '')}_{max_steps}.mp4"


def get_parameters(env, alpha=0.001, gamma=0.99):
    return {
        # NN input and output
        "nr_actions": env.action_space.n,
        "nr_input_features": env.observation_space.shape,
        # Hyperparameters
        "alpha": alpha,  # learning rate
        "gamma": gamma,  # discount factor
    }


# Setup file names
training_episodes = 1
max_steps = 100
rooms_dir = "9_9_4"

room_files = get_room_files(rooms_dir)
room_file = room_files[1]
room_path = get_room_path(rooms_dir, room_file)
movie_file_name = get_movie_file_path(room_file, max_steps)

# Domain setup
env = rooms.load_env(room_path, movie_file_name, max_steps)

# params
params = get_parameters(env, alpha=0.001, gamma=0.99)
print(f"params: {params}")

# Agent setup
main_agent = a2c.A2CLearner(params)
returns = [episode(env, main_agent, i) for i in range(training_episodes)]

x_data = range(training_episodes)
y_data = returns

create_plot("Progress", x_data, "episode", y_data, "discounted return")

env.save_video()
