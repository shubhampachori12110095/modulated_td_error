import random

from environments.card_sorting import CardSortingEnv
from learners.dqn import DQN
from util.experiment import Experiment

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
random.seed(0)

n_features = 4
n_classes = 4


if __name__ == '__main__':

    # enumerate params for a conventional tabular learner and the learner based on the new rule
    for agent_params in [
        {
            'n_inputs': [n_features], 'n_outputs': [n_classes], 'batch_size': [10], 'epsilon': [0.1],
            'replay_buffer_size': [10],
            'hidden_layer_sizes': [(20,)], 'update_frequency': [1], 'lr': [0.005, 0.01, 0.05, 0.1], 'gamma': [0],
            'modulated_td_error': [False]
        },
        {
            'n_inputs': [n_features], 'n_outputs': [n_classes], 'batch_size': [10], 'epsilon': [0.1],
            'replay_buffer_size': [10],
            'hidden_layer_sizes': [(20,)], 'update_frequency': [1], 'lr': [0.005, 0.01, 0.05, 0.1], 'gamma': [0],
            'modulated_td_error': [True]
        }
    ]:
        # create the "Card Sorting" task
        env = CardSortingEnv(n_features=4, n_classes=4, samples_before_change=100, show_current_ordering=False)

        # run experiment
        experiment = Experiment(environment=env, agent=DQN, params=agent_params, param_search_steps=2000)
        experiment.step(4000)

        plt.plot(experiment.reward_history.cumsum())

    plt.title('cumulative reward on "card sorting" task')
    plt.legend(['conventional deep-q net', 'new rule'])
    plt.ylabel('cumulative reward obtained')
    plt.xlabel('steps')
    plt.show()
