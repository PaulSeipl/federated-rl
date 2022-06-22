from pathlib import Path
import random
from config import (
    AGGREATION_TYPES,
    PLOT_PATH_SOLO,
    ROTATE_MAP,
    SAVE_PLOTS,
    PLOT_PATH,
    TEST_DIR,
    TEST_EPISODES,
    AGGREATION_TYPE,
    TEST_STOCHASTIC,
)
from plots.helper import create_main_plot, get_plot_file_path, save_test_data


def episode(env, agent, nr_episode=0, increment=0, isTraining=True):
    state = env.reset()
    policy_func = (
        agent.policy if isTraining or TEST_STOCHASTIC else agent.policy_deterministic
    )
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    while not done:
        action = policy_func(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        if isTraining:
            agent.update(state, action, reward, next_state, done)
        state = next_state
        discounted_return += reward * (discount_factor ** time_step)
        time_step += 1
    print(agent.name, ", ", nr_episode + increment, ":", discounted_return)
    return discounted_return


def train(
    working_intervals,
    training_episodes,
    worker_envs,
    worker_agents,
    main_agent,
    test_envs,
):
    worker_returns = [(worker_agent.name, []) for worker_agent in worker_agents]
    for interval in range(working_intervals):
        worker_state_dicts = []
        for index, (env, worker) in enumerate(zip(worker_envs, worker_agents)):
            # initialize each agent with state_dict from main agent
            worker.load_state_dict(main_agent.get_state_dict_copy())

            # run {training_episodes} episodes for each agent
            for i in range(training_episodes):
                worker_returns[index][1].append(
                    episode(env, worker, i, interval * training_episodes)
                )
                # rotate map after every episode
                if ROTATE_MAP:
                    env.rotate_map_random()

            # safe worker state_dict
            worker_state_dicts.append(worker.get_state_dict_copy())

            # rotate workers env after every aggregation
            # if index < working_intervals:
            #     env.rotate_map_random()

        main_state_dict = {}
        # set main state dict
        if AGGREATION_TYPE == AGGREATION_TYPES[1]:
            main_state_dict = average(
                main_agent.get_state_dict().keys(), worker_state_dicts
            )

        if AGGREATION_TYPE == AGGREATION_TYPES[2]:
            main_state_dict = zero_weighted_aggregate(
                main_agent.get_state_dict().keys(),
                worker_state_dicts,
                worker_returns,
                training_episodes,
            )

        if AGGREATION_TYPE == AGGREATION_TYPES[3]:
            main_state_dict = positive_weighted_aggregate(
                main_agent.get_state_dict().keys(),
                worker_state_dicts,
                worker_returns,
                training_episodes,
            )

        main_agent.load_state_dict(main_state_dict)

        if interval != working_intervals - 1:
            # test main agent
            if TEST_STOCHASTIC:
                test_agent_stochastic(main_agent, test_envs, interval + 1)
            else:
                test_agent_deterministic(main_agent, test_envs, interval + 1)

    return worker_returns


def solo_train(
    agent,
    envs,
    test_envs,
    episodes,
):
    agent_returns = []
    for i in range(episodes):
        print("training episode: ", i)
        # random env
        rand_env = envs[random.randint(0, len(envs) - 1)]
        # run episode
        agent_returns.append(episode(rand_env, agent, i))
        # random rotate of map
        rand_env.rotate_map_random()

        # test every 100 episodes
        if (i + 1) % 100 == 0:
            if TEST_STOCHASTIC:
                test_solo_agent_stochastic(agent, test_envs, i + 1)
            else:
                pass  # test solo determenistic

    return agent_returns


def test_solo_agent_stochastic(agent, envs, episodes):
    agent.a2c_net.eval()
    returns = []
    for env in envs:
        agent.name = env.name
        returns.append(
            [episode(env, agent, i, isTraining=False) for i in range(TEST_EPISODES)]
        )
    agent.name = "main_agent"
    agent.a2c_net.train()

    if SAVE_PLOTS:
        Path(PLOT_PATH_SOLO).mkdir(parents=True, exist_ok=True)
        x_data = [env.name for env in envs]
        y_data = []
        for data in returns:
            y_data.append(sum(data) / len(data))

        create_main_plot(
            f"Progress: {agent.name} after {episodes} episodes on {TEST_DIR} test rooms",
            x_data,
            "test room",
            y_data,
            f"average discounted return of {TEST_EPISODES} episodes",
            get_plot_file_path(
                f"{agent.name}_after {episodes} episodes",
                PLOT_PATH_SOLO,
            ),
        )
        save_test_data(x_data, returns, PLOT_PATH_SOLO, episodes)


def test_agent_stochastic(agent, envs, updates):
    update_str = "update" if updates == 1 else "updates"
    # run test with main_agent
    agent.a2c_net.eval()
    returns = []
    for env in envs:
        agent.name = env.name
        returns.append(
            [episode(env, agent, i, isTraining=False) for i in range(TEST_EPISODES)]
        )
    agent.name = "main_agent"
    agent.a2c_net.train()

    if SAVE_PLOTS:
        Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)
        x_data = [env.name for env in envs]
        y_data = []
        for data in returns:
            y_data.append(sum(data) / len(data))

        create_main_plot(
            f"Progress: {agent.name} after {updates} {update_str} on {TEST_DIR} test rooms",
            x_data,
            "test room",
            y_data,
            f"average discounted return of {TEST_EPISODES} episodes",
            get_plot_file_path(
                f"{agent.name}_after{updates}{update_str}",
                PLOT_PATH,
            ),
        )
        save_test_data(x_data, returns, PLOT_PATH, updates)


def test_agent_deterministic(agent, envs, updates=0):
    update_str = "update" if updates == 1 else "updates"
    # run test with main_agent
    agent.a2c_net.eval()  # TODO STOCHASTIC: Add test episodes
    returns = [episode(env, agent, 0, isTraining=False) for env in envs]
    agent.a2c_net.train()

    if SAVE_PLOTS:
        Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)
        x_data = [env.name for env in envs]
        y_data = returns

        create_main_plot(
            f"Progress: {agent.name} after {updates} {update_str} on {TEST_DIR} test rooms",
            x_data,
            "test room",
            y_data,
            "discounted return",
            get_plot_file_path(
                f"{agent.name}_after{updates}{update_str}",
                PLOT_PATH,
            ),
        )


def average(main_agent_state_dict_keys, worker_state_dicts):
    # calculate mean of the state_dicts (calculation manually checked, works)
    mean_state_dict = {}
    for key in main_agent_state_dict_keys:
        mean_state_dict[key] = sum(
            [worker_state_dict[key] for worker_state_dict in worker_state_dicts]
        ) / len(worker_state_dicts)

    return mean_state_dict


def get_weighted_zero_distribution(worker_returns, separator=0):
    # count zeros
    zeros_count = [
        worker_return[-separator:].count(0) for _, worker_return in worker_returns
    ]
    print(f"zeros_count: \n{zeros_count}")
    # if every worker reached in every episode the goal
    if sum(zeros_count) == 0:
        return [1 / len(worker_returns) for _ in worker_returns]

    return [zero_count / sum(zeros_count) for zero_count in zeros_count]


# je öfter der Agent das Ziel nicht erreicht hat, desto mehr werden seine weights wahrgenommen/miteinberechnet
def zero_weighted_aggregate(
    main_agent_state_dict_keys, worker_state_dicts, worker_returns, separator=0
):
    # get distribution (adds up to 1)
    weighted_distribution = get_weighted_zero_distribution(worker_returns, separator)
    print(f"weighted_distribution: \n{weighted_distribution}")
    mean_state_dict = {}
    for key in main_agent_state_dict_keys:
        mean_state_dict[key] = sum(
            [
                worker_state_dict[key] * multiplicator
                for worker_state_dict, multiplicator in zip(
                    worker_state_dicts, weighted_distribution
                )
            ]
        )

    return mean_state_dict


def positive_weighted_aggregate(
    main_agent_state_dict_keys, worker_state_dicts, worker_returns, separator=0
):
    # get distribution (adds up to 1)
    # count numbers bigger then 0
    trues_count = [
        len(list(filter(bool, worker_return[-separator:])))
        for _, worker_return in worker_returns
    ]
    print("true_count: ", trues_count)
    weighted_distribution = [
        true_count / sum(trues_count) for true_count in trues_count
    ]

    print(f"weighted_distribution: \n{weighted_distribution}")

    mean_state_dict = {}
    for key in main_agent_state_dict_keys:
        mean_state_dict[key] = sum(
            [
                worker_state_dict[key] * multiplicator
                for worker_state_dict, multiplicator in zip(
                    worker_state_dicts, weighted_distribution
                )
            ]
        )

    return mean_state_dict
