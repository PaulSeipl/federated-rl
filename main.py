from copy import deepcopy
from numpy.core.fromnumeric import mean
from src import rooms, a2c
from plots.helper import (
    create_main_plot,
    create_worker_plot,
    get_plot_file_path,
    save_worker_data,
    save_test_data,
)
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


def initialize_main_agent(test_env):
    return a2c.A2CLearner(
        get_parameters(test_env, "main_agent", alpha=0.001, gamma=0.99)
    )


def get_test_envs(rooms_dir, movie_path):
    test_rooms_dir = f"{rooms_dir}/test/{TEST_DIR}"
    test_room_files = layouts.get_room_files(test_rooms_dir)
    print(test_room_files)
    test_envs = []
    # get every test room file
    for test_room_file in test_room_files:
        name = test_room_file.replace(".txt", "_0")

        test_env = rooms.load_env(
            layouts.get_room_path(test_rooms_dir, test_room_file),
            get_movie_file_path(name, movie_path),
            MAX_STEPS,
            room_name=name,
        )

        test_envs.append(test_env)
        # rotate every test room env and save it as own test room env
        for i in range(1, 4):
            temp_env = deepcopy(test_env)
            for _ in range(i):
                temp_env.rotate_map()
            temp_env.name = temp_env.name.replace("_0", f"_{i}")
            test_envs.append(temp_env)
    return test_envs


def main():
    test_envs = get_test_envs(ROOMS_DIR, MOVIE_PATH)
    main_agent = initialize_main_agent(test_envs[0])

    room_files = layouts.get_room_files(ROOMS_DIR)
    print(f"room_files: {room_files}")

    # Domain and worker agents setup
    worker_envs = []
    parameterList = []
    worker_agents = []
    for room_file in room_files:
        if AGENT_PER_PERMUTATION:
            name = room_file.replace(".txt", "_0")
        else:
            name = room_file.replace(".txt", "")

        worker_env = rooms.load_env(
            layouts.get_room_path(ROOMS_DIR, room_file),
            get_movie_file_path(name, MOVIE_PATH),
            MAX_STEPS,
            room_name=name,
        )
        worker_envs.append(worker_env)

        if AGENT_PER_PERMUTATION:
            # rotate every test room env and save it as own test room env
            for i in range(1, 4):
                temp_env = deepcopy(worker_env)
                for _ in range(i):
                    temp_env.rotate_map()
                temp_env.name = temp_env.name.replace("_0", f"_{i}")
                worker_envs.append(temp_env)

    print(f"all rooms: {[worker_env.name for worker_env in worker_envs]}")
    for worker_env in worker_envs:
        parameter = get_parameters(worker_env, worker_env.name, alpha=0.001, gamma=0.99)

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
        test_envs,
    )

    # run test with main_agent
    main_agent.a2c_net.eval()
    returns = []
    if TEST_STOCHASTIC:
        for worker_env in test_envs:
            main_agent.name = worker_env.name
            returns.append(
                [
                    episode(worker_env, main_agent, i, isTraining=False)
                    for i in range(TEST_EPISODES)
                ]
            )
        main_agent.name = "main_agent"
    else:
        returns = [
            episode(test_env, main_agent, 0, isTraining=False) for test_env in test_envs
        ]

    if SAVE_PLOTS:
        # create plot folder
        Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)

        x_data = [test_env.name for test_env in test_envs]
        if TEST_STOCHASTIC:
            y_data = []
            for data in returns:
                y_data.append(sum(data) / len(data))

            create_main_plot(
                f"Evaluation: {main_agent.name}",
                x_data,
                "test room",
                y_data,
                f"average discounted return of {TEST_EPISODES} episodes",
                get_plot_file_path(
                    f"{main_agent.name}",
                    PLOT_PATH,
                ),
            )
            save_test_data(x_data, returns)
        else:
            y_data = returns
            create_main_plot(
                f"Evaluation: {main_agent.name}",
                x_data,
                "test room",
                y_data,
                "discounted return",
                get_plot_file_path(
                    f"{main_agent.name}",
                    PLOT_PATH,
                ),
            )

        for (name, worker_return) in worker_returns:
            save_worker_data(worker_return, name)

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

    if SAVE_VIDEOS:
        # create movie folder
        Path(MOVIE_PATH).mkdir(parents=True, exist_ok=True)
        for worker_env in worker_envs + test_envs:
            worker_env.save_video()

    if SAVE_MODELS:
        Path(f"./models/{ROOMS_DIR}").mkdir(parents=True, exist_ok=True)
        main_agent.save_net(f"{MODEL_PATH}_{main_agent.name}")
        main_agent.save_state_dict(f"{MODEL_PATH}_{main_agent.name}")


if __name__ == "__main__":
    main()
