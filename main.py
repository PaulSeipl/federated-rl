from numpy.core.fromnumeric import mean
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
    print(agent.name, ", ", nr_episode, ":", discounted_return)
    return discounted_return


def create_plot(title, x_data, x_label, y_data, y_label):
    plot.close()
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
        # "id"
        "name": f"Progress: {env.movie_filename.replace('.mp4', '')}",
    }


def initialize_main_agent(max_steps):
    env = rooms.load_env(
        "layouts/9_9_4/test/rooms_9_9_4_t0.txt",
        "movies/rooms_9_9_4_t0.mp4",
        max_steps,
    )
    agent = a2c.A2CLearner(get_parameters(env, alpha=0.001, gamma=0.99))
    # agent.a2c_net.train(False)
    return env, agent


def main():
    # Setup file names
    test_episodes = 1
    working_intervals = 1
    max_steps = 100
    rooms_dir = "9_9_4"
    test_env, main_agent = initialize_main_agent(max_steps)

    room_files = get_room_files(rooms_dir)

    # Domain setup
    envs = [
        rooms.load_env(
            get_room_path(rooms_dir, room_file),
            get_movie_file_path(room_file, max_steps),
            max_steps,
        )
        for room_file in room_files
    ]

    # params
    print(f"params: {get_parameters(envs[0], alpha=0.001, gamma=0.99)}")

    # multiple agents
    worker_agents = [
        a2c.A2CLearner(get_parameters(env, alpha=0.001, gamma=0.99)) for env in envs
    ]
    # run x "training_episodes" for every agent "working_intervals" times
    for _ in range(working_intervals):
        worker_state_dicts = []
        for (env, worker) in zip(envs, worker_agents):
            # initialize each agent with state_dict from main agent
            worker.load_state_dict(main_agent.get_state_dict_copy())

            # run {training_episodes} episodes for each agent
            [episode(env, worker, i) for i in range(test_episodes)]

            # safe worker state_dict
            worker_state_dicts.append(worker.get_state_dict_copy())

        # calculate mean of the state_dicts
        mean_state_dict = {}
        for key in main_agent.get_state_dict().keys():
            mean_state_dict[key] = sum(
                [worker_state_dict[key] for worker_state_dict in worker_state_dicts]
            ) / len(worker_state_dicts)

        # set main_agent state dict
        main_agent.load_state_dict(mean_state_dict)

    test_episodes = 100
    # run test with main_agent
    # main_agent.a2c_net.eval()
    returns = [episode(test_env, main_agent, i) for i in range(test_episodes)]

    x_data = range(test_episodes)
    y_data = returns

    create_plot(
        f"Progress: {test_env.movie_filename.replace('.mp4', '')}",
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
