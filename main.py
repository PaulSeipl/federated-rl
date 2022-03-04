from numpy.core.fromnumeric import mean
from src import rooms, a2c
from plots.helper import create_main_plot, create_worker_plot, get_plot_file_path
import layouts.helper as layouts
from src.train import train, episode
from pathlib import Path
from config import *


def get_movie_file_path(name, movie_path):
    return f"{movie_path}/{name}.mp4"


def get_parameters(env, name, alpha=0.001, gamma=0.99):
    return {
        # NN input and output
        "nr_actions": env.action_space.n,
        "nr_input_features": env.observation_space.shape,
        # Hyperparameters
        "alpha": alpha,  # learning rate
        "gamma": gamma,  # discount factor
        # update timer
        "max_done_counter": MAX_DONE_COUNTER,
        # "id"
        "name": name,
    }


def initialize_main_agent(max_steps, rooms_dir, movie_path):
    name = "t0"
    env = rooms.load_env(
        f"layouts/{rooms_dir}/test/t0.txt",
        get_movie_file_path(name, movie_path),
        max_steps,
    )
    agent = a2c.A2CLearner(get_parameters(env, "t0", alpha=0.001, gamma=0.99))
    # agent.a2c_net.train(False)
    return env, agent


def main():
    test_env, main_agent = initialize_main_agent(MAX_STEPS, ROOMS_DIR, MOVIE_PATH)

    room_files = layouts.get_room_files(ROOMS_DIR)
    print(f"room_files: {room_files}")

    # Domain and worker agents setup
    worker_envs = []
    parameterList = []
    worker_agents = []
    for room_file in room_files:
        name = room_file.replace(".txt", "")

        env = rooms.load_env(
            layouts.get_room_path(ROOMS_DIR, room_file),
            get_movie_file_path(name, MOVIE_PATH),
            MAX_STEPS,
        )
        worker_envs.append(env)

        parameter = get_parameters(env, name, alpha=0.001, gamma=0.99)

        parameterList.append(parameter)

        agent = a2c.A2CLearner(parameter)

        worker_agents.append(agent)

    # params
    print(f"params: {parameterList[0]}")

    # run x "training_episodes" for every agent "working_intervals" times
    worker_returns = train(
        WORKING_INTERVALS,
        TRAINING_EPISODES,
        worker_envs,
        worker_agents,
        main_agent,
        test_env,
    )

    # run test with main_agent
    main_agent.a2c_net.eval()
    returns = [episode(test_env, main_agent, i) for i in range(TEST_EPISODES)]

    if SAVE_PLOTS:
        # create plot folder
        Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)

        for (name, worker_return) in worker_returns:
            x_data = range(len(worker_return))
            y_data = worker_return
            create_worker_plot(
                f"Progress: Room {name}, {TRAINING_EPISODES*WORKING_INTERVALS} Trainingepisoden, Update nach {TRAINING_EPISODES}",
                x_data,
                "episode",
                y_data,
                "discounted return",
                WORKING_INTERVALS,
                TRAINING_EPISODES,
                get_plot_file_path(name, PLOT_PATH),
            )

        x_data = range(TEST_EPISODES)
        y_data = returns

        create_main_plot(
            f"Progress: {main_agent.name}",
            x_data,
            "episode",
            y_data,
            "discounted return",
            get_plot_file_path(
                main_agent.name,
                PLOT_PATH,
            ),
        )

    if SAVE_VIDEOS:
        # create movie folder
        Path(MOVIE_PATH).mkdir(parents=True, exist_ok=True)
        for env in worker_envs:
            env.save_video()
        test_env.save_video()

    if SAVE_MODELS:
        main_agent.save_net(
            f"{ROOMS_DIR}_tr{TRAINING_EPISODES}_x{WORKING_INTERVALS}_t{TEST_EPISODES}_{main_agent.name}"
        )


if __name__ == "__main__":
    main()
