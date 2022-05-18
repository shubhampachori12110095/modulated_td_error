import itertools
import multiprocessing
from copy import deepcopy

import numpy as np


class ParamEvaluator:

    def __init__(self, environment, agent_class, param_search_steps=1000):
        self.environment = deepcopy(environment)
        self.agent_class = agent_class
        self.param_search_steps = param_search_steps

    def __call__(self, params):
        experiment = Experiment(self.environment, self.agent_class(**params))
        experiment.step(self.param_search_steps)
        return experiment.reward_history.sum()


def optimize_params(environment, agent_class, params, param_search_steps=1000):
    combinations = [dict(zip(list(params), combination)) for combination in itertools.product(*params.values())]
    if len(combinations) == 1:
        return agent_class(**combinations[0])
    else:
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        rewards = pool.map(ParamEvaluator(environment, agent_class, param_search_steps=param_search_steps), combinations)
        pool.close()
        best_idx = np.argmax(rewards)
        print(f'{agent_class} best params:', combinations[best_idx])
        return agent_class(**combinations[best_idx])


class Experiment:
    """
    Utility class for running experiments
    """

    def __init__(self, environment, agent, params=None, param_search_steps=1000, sensors=tuple()):
        """
        :param environment: an environment instance
        :param agent: a learner instance, or learner class. If a class, params must be provided
        :param params: A dict mapping learner parameters to lists of values. All combinations will be tried, and the
        best used in the experiment
        :param param_search_steps: number of steps to use during parameter searches
        :param sensors: a list of callables that accept the environment and agent as arguments. The values they return
        will be saved and available through the .sensor_readings member
        """

        self.environment = deepcopy(environment)
        self.agent = deepcopy(agent)
        self.sensors = sensors
        self.sensor_readings = {sensor: [] for sensor in sensors}

        if params is not None:
            self.agent = optimize_params(environment, agent, params, param_search_steps)
        self.reward_history = np.zeros(0)
        self.action_history = []

        self.state = environment.reset()

    def step(self, n):
        this_rewards = np.zeros(n)
        these_actions = [None] * n
        these_readings = {sensor: [None] * n for sensor in self.sensors}

        for step in range(n):  #tqdm(range(n)):
            action = self.agent.select_action(self.state)
            new_state, reward, done, _ = self.environment.step(action)
            self.agent.update(self.state, action, reward, new_state, done)
            this_rewards[step] = reward
            these_actions[step] = action

            self.state = new_state
            for sensor in these_readings:
                these_readings[sensor][step] = sensor(self.environment, self.agent)

        self.reward_history = np.append(self.reward_history, this_rewards)
        for sensor in self.sensor_readings:
            self.sensor_readings[sensor] += these_readings[sensor]
