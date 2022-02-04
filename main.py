from numpy.core.fromnumeric import mean
from src import rooms, a2c
from plots.helper import create_main_plot, create_worker_plot
import layouts.helper as layouts
from src.train import train, episode


def get_movie_file_path(room_file, max_steps):
    return f"movies/{room_file.replace('.txt', '')}_{max_steps}.mp4"


def get_plot_file_path(name, max_steps):
    return f"plots/{name}_{max_steps}"


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
        f"layouts/{rooms_dir}/test/t0.txt",
        f"movies/t0.mp4",
        max_steps,
    )
    agent = a2c.A2CLearner(get_parameters(env, "t0", alpha=0.001, gamma=0.99))
    # agent.a2c_net.train(False)
    return env, agent


def main():
    # Setup file names
    training_episodes = 500
    working_intervals = 20
    test_episodes = 100
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
    worker_returns = train(
        working_intervals, training_episodes, worker_envs, worker_agents, main_agent
    )

    for (name, worker_return) in worker_returns:
        x_data = range(len(worker_return))
        y_data = worker_return
        plot_name = f"tr{training_episodes}_x{working_intervals}"
        create_worker_plot(
            f"Progress: Room {name}, {training_episodes*working_intervals} Trainingepisoden, Update nach {training_episodes}",
            x_data,
            "episode",
            y_data,
            "discounted return",
            working_intervals,
            training_episodes,
            get_plot_file_path(
                name,
                plot_name,
            ),
        )

    # run test with main_agent
    # main_agent.a2c_net.eval()
    returns = [episode(test_env, main_agent, i) for i in range(test_episodes)]

    x_data = range(test_episodes)
    y_data = returns

    create_main_plot(
        f"Progress: {main_agent.name}",
        x_data,
        "episode",
        y_data,
        "discounted return",
        get_plot_file_path(
            main_agent.name,
            f"tr{training_episodes}_x{working_intervals}_t{test_episodes}",
        ),
    )

    # test_env.save_video()
    main_agent.save_net(
        f"{rooms_dir}_tr{training_episodes}_x{working_intervals}_t{test_episodes}_{main_agent.name}"
    )
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
