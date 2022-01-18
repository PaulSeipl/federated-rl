from numpy.core.fromnumeric import mean
from src import rooms, a2c
import matplotlib.pyplot as plot
import layouts.helper as layouts
from src.train import train, episode


def create_plot(title, x_data, x_label, y_data, y_label):
    plot.close()
    plot.plot(x_data, y_data)
    plot.title(title)
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.show()


def get_movie_file_path(room_file, max_steps):
    return f"movies/{room_file.replace('.txt', '')}_{max_steps}.mp4"


def get_parameters(env, name, alpha=0.001, gamma=0.99):
    return {
        # NN input and output
        "nr_actions": env.action_space.n,
        "nr_input_features": env.observation_space.shape,
        # Hyperparameters
        "alpha": alpha,  # learning rate
        "gamma": gamma,  # discount factor
        # "id"
        "name": name,
    }


def initialize_main_agent(max_steps, rooms_dir):
    env = rooms.load_env(
        f"layouts/{rooms_dir}/test/rooms_{rooms_dir}_t0.txt",
        f"movies/rooms_{rooms_dir}_t0.mp4",
        max_steps,
    )
    agent = a2c.A2CLearner(
        get_parameters(env, f"{rooms_dir}_t0", alpha=0.001, gamma=0.99)
    )
    # agent.a2c_net.train(False)
    return env, agent


def main():
    # Setup file names
    training_episodes = 1
    working_intervals = 1
    max_steps = 100
    rooms_dir = "9_9_4"
    test_env, main_agent = initialize_main_agent(max_steps, rooms_dir)

    room_files = layouts.get_room_files(rooms_dir)
    print(f"room_files: {room_files}")

    # Domain and worker agents setup
    worker_envs = []
    parameterList = []
    worker_agents = []
    for room_file in room_files:
        env = rooms.load_env(
            layouts.get_room_path(rooms_dir, room_file),
            get_movie_file_path(room_file, max_steps),
            max_steps,
        )
        worker_envs.append(env)

        parameter = get_parameters(
            env, room_file.replace(".txt", ""), alpha=0.001, gamma=0.99
        )

        parameterList.append(parameter)

        agent = a2c.A2CLearner(parameter)

        worker_agents.append(agent)

    # params
    print(f"params: {parameterList[0]}")

    # run x "training_episodes" for every agent "working_intervals" times
    train(working_intervals, training_episodes, worker_envs, worker_agents, main_agent)

    test_episodes = 100
    # run test with main_agent
    # main_agent.a2c_net.eval()
    returns = [episode(test_env, main_agent, i) for i in range(test_episodes)]

    x_data = range(test_episodes)
    y_data = returns

    create_plot(
        f"Progress: {main_agent.name}",
        x_data,
        "episode",
        y_data,
        "discounted return",
    )

    # test_env.save_video()
    main_agent.save_net("main_agent")
    # returns = [episode(env, agent, i) for i in range(training_episodes)]
    # x_data = range(training_episodes)
    # y_data = returns
    # create_plot(
    #     f"Progress: {env.movie_filename.replace('.mp4', '')}",
    #     x_data,
    #     "episode",
    #     y_data,
    #     "discounted return",
    # )
    # env.save_video()


if __name__ == "__main__":
    main()
