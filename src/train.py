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


def train(working_intervals, training_episodes, worker_envs, worker_agents, main_agent):
    for _ in range(working_intervals):
        worker_state_dicts = []
        for (env, worker) in zip(worker_envs, worker_agents):
            # initialize each agent with state_dict from main agent
            worker.load_state_dict(main_agent.get_state_dict_copy())

            # run {training_episodes} episodes for each agent
            [episode(env, worker, i) for i in range(training_episodes)]

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
