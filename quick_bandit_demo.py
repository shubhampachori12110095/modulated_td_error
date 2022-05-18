import random

from environments.n_armed_bandit import ChangingNArmedBanditEnv
from learners.tabular import TabularLearner
from util.experiment import Experiment

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
random.seed(0)

n = 7


if __name__ == '__main__':

    # enumerate params for a conventional tabular learner and the learner based on the new rule
    for agent_params in [
        {'action_list': [np.arange(n)], 'gamma': [0.75], 'alpha': np.arange(0.1, 2, 0.2), 'default_value': [1],
         'softmax_temperature': [0.25, 1], 'modulated_td_error': [False]},
        {'action_list': [np.arange(n)], 'gamma': [0.75], 'alpha': np.arange(0.1, 2, 0.2), 'default_value': [1],
         'softmax_temperature': [0.25, 1], 'modulated_td_error': [True]},
    ]:
        # create the n-armed bandit task with one high-reward arm, one no-reward, and the rest ~U(0.25, 0.75)
        p = np.random.rand(n) * 0.5 + 0.25
        p[-1] = 0
        p[0] = 0.9
        env = ChangingNArmedBanditEnv(p=p, rotation_interval=100)

        # run experiment
        experiment = Experiment(environment=env, agent=TabularLearner, params=agent_params, param_search_steps=2000)
        experiment.step(4000)

        plt.plot(experiment.reward_history.cumsum())

    plt.title('cumulative reward on changing n-armed bandit')
    plt.legend(['conventional q learning', 'new rule'])
    plt.ylabel('cumulative reward obtained')
    plt.xlabel('steps')
    plt.show()