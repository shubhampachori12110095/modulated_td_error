import copy
import random

import numpy as np
import torch
from torch import nn

from learners.abstract import ReinforcementLearner



def temperature_softmax(x, dim=0, t=1.0):
    sm = torch.nn.Softmax(dim=dim)
    return sm(x/t)


def build_network(n_inputs, n_outputs, layer_sizes, activation=nn.Tanh):
    layer_sizes = [n_inputs] + list(layer_sizes) + [n_outputs]
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i < len(layer_sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class TransitionMemory:

    def __init__(self, capacity, state_size, device):
        self.index = -1
        self.size = 0
        self.capacity = capacity
        self.states = torch.zeros((capacity, state_size), device=device)
        self.actions = torch.zeros(capacity, device=device, dtype=torch.long)
        self.rewards = torch.zeros(capacity, device=device)
        self.new_states = torch.zeros((capacity, state_size), device=device)
        self.done = torch.zeros(capacity, device=device)

    def add(self, state, action, reward, new_state, done):
        self.index = (self.index + 1) % self.capacity

        self.states[self.index, :] = torch.tensor(state)
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.new_states[self.index, :] = torch.tensor(new_state)
        self.done[self.index] = torch.tensor(float(done))

        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        n = min(n, self.size)

        idx = np.random.choice(self.size, n, replace=False)
        return self.states[idx, :], self.actions[idx], self.rewards[idx], self.new_states[idx, :], self.done[idx]

    def last(self):
        return self.states[self.index, :], self.actions[self.index], self.rewards[self.index], self.new_states[self.index], self.done[self.index]


class DQN(ReinforcementLearner):
    """
    A deep Q network that optionally implements the rule from our paper:
     Brain-Inspired modulation of reward-prediction error improves reinforcement learning adaptation to environmental
     change
    """

    # thanks to https://towardsdatascience.com/deep-q-learning-for-the-cartpole-44d761085c2f for getting started

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __init__(self, n_inputs, n_outputs, batch_size, hidden_layer_sizes, replay_buffer_size, update_frequency=10,
                 lr=1e-3, sync_frequency=5,
                 gamma=0.95, epsilon=0.1,
                 modulated_td_error=False,
                 softmax_temp=1.0,
                 seed=42):
        """
        :param n_inputs: length of input vector
        :param n_outputs: length of output vector
        :param batch_size: number of experiences to sample from the replay buffer at each learning step
        :param hidden_layer_sizes: list of hidden layer sizes
        :param replay_buffer_size: number of experiences to keep in the replay buffer
        :param update_frequency: number of steps before updating models
        :param lr: learning rate
        :param sync_frequency: number of steps to run between syncing the policy and value networks
        :param gamma: discount factor
        :param epsilon: parameter for e-greedy action sampling
        :param modulated_td_error: if True, will use the new RL rule from the paper
        :param softmax_temp: softmax tempurature for use in new RL rule
        :param seed: random seed
        """
        torch.manual_seed(seed)

        # instantiate networks
        self.policy_net = build_network(n_inputs, n_outputs, hidden_layer_sizes, activation=nn.Tanh)
        self.value_net = copy.deepcopy(self.policy_net)
        self.policy_net.to(self.device)
        self.value_net.to(self.device)

        # instantiate loss and optimizer
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        # instantiate experience memory
        self.transition_memory = TransitionMemory(capacity=replay_buffer_size, state_size=n_inputs, device=self.device)

        # store other params
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.n_outputs = n_outputs
        self.sync_frequency = sync_frequency
        self.sync_counter = 0
        self.gamma = torch.tensor(gamma, device=self.device)
        self.epsilon = epsilon
        self.update_counter = 0
        self.modulated_td_error = modulated_td_error
        self.softmax_temp = softmax_temp

    def select_action(self, state):
        if random.random() < self.epsilon:
            return np.random.choice(self.n_outputs)
        with torch.no_grad():
            max_q, index = self.policy_net(torch.tensor(state.astype(np.float32), device=self.device)).max(0)
        return index.item()

    def update(self, state, action, reward, new_state, done):

        if float(reward) not in [-1, 0, 1]:
            raise ValueError('current implementation of DQN requires rewards of 1, 0, or -1')

        self.transition_memory.add(state, action, reward, new_state, done)
        self.update_counter += 1
        if self.update_counter % self.update_frequency == 0:

            # sync value and policy networks
            self.sync_counter += 1
            if self.sync_counter % self.sync_frequency == 0:
                self.value_net.load_state_dict(self.policy_net.state_dict())

            s, a, r, ns, d = self.transition_memory.sample(self.batch_size)

            # get policy network's current value estimates
            state_action_values = self.policy_net(s)

            # get target value estimates, based on actual rewards and value net's predictions of next-state value
            with torch.no_grad():
                new_state_value, _ = self.value_net(ns).max(1)
            target_action_value = r + self.gamma * new_state_value * (1 - d)
            target_values = state_action_values.clone().detach()
            if self.modulated_td_error:
                probabilities = temperature_softmax(state_action_values, dim=1, t=self.softmax_temp)
                target_action_value *= probabilities[np.arange(target_values.shape[0]), a]
            target_values[np.arange(target_values.shape[0]), a] = target_action_value
            # todo: fix the update - right now it only works for immediate rewards of 1 and -1

            # optimize loss
            loss = self.loss_fn(state_action_values, target_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()
