from pathlib import Path
from config import (
    AGGREATION_TYPES,
    SAVE_PLOTS,
    PLOT_PATH,
    TEST_EPISODES,
    AGGREATION_TYPE,
)
from plots.helper import create_main_plot, get_plot_file_path


def episode(env, agent, nr_episode=0, increment=0):
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
    print(agent.name, ", ", nr_episode + increment, ":", discounted_return)
    return discounted_return


def train(
    working_intervals,
    training_episodes,
    worker_envs,
    worker_agents,
    main_agent,
    main_env,
):
    worker_returns = [(worker_agent.name, []) for worker_agent in worker_agents]
    for interval in range(working_intervals):
        worker_state_dicts = []
        for index, (env, worker) in enumerate(zip(worker_envs, worker_agents)):
            # initialize each agent with state_dict from main agent
            worker.load_state_dict(main_agent.get_state_dict_copy())

            # run {training_episodes} episodes for each agent

            [
                worker_returns[index][1].append(
                    episode(env, worker, i, interval * training_episodes)
                )
                for i in range(training_episodes)
            ]

            # safe worker state_dict
            worker_state_dicts.append(worker.get_state_dict_copy())

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
            test_agent(main_agent, main_env, interval + 1)  # TODO create random env

    return worker_returns


def test_agent(agent, env, updates=0):
    update_str = "update" if updates == 1 else "updates"
    # run test with main_agent
    agent.a2c_net.eval()
    returns = [episode(env, agent, i) for i in range(TEST_EPISODES)]
    agent.a2c_net.train()

    if SAVE_PLOTS:
        Path(PLOT_PATH).mkdir(parents=True, exist_ok=True)
        x_data = range(TEST_EPISODES)
        y_data = returns

        create_main_plot(
            f"Progress: {agent.name} after {updates} {update_str}",
            x_data,
            "episode",
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
