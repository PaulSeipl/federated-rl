from pathlib import Path
from config import SAVE_PLOTS
from plots.helper import create_main_plot, get_plot_file_path
from config import PLOT_PATH, TEST_EPISODES


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

        # calculate mean of the state_dicts
        mean_state_dict = {}
        for key in main_agent.get_state_dict().keys():
            mean_state_dict[key] = sum(
                [worker_state_dict[key] for worker_state_dict in worker_state_dicts]
            ) / len(worker_state_dicts)

        # set main_agent state dict
        main_agent.load_state_dict(mean_state_dict)

        if interval != working_intervals:
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
