from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np
from gym.spaces import Box, Discrete


class CardSortingEnv:
    """
        Simulates a task similar to the Wisconsin Card Sorting Test using a multiclass classification task.
        Normally distributed clusters of points are created in n-dimensional space and assigned to each of k classes.
        The agent is rewarded when it correctly matches a randomly-drawn point to its current class, but the classes are
        periodically scrambled (such that all the points previously assigned to class “0” now belong to class “2”, for
        example)
    """

    def __init__(self, n_classes, n_features, samples_before_change, callback_on_change=None, show_current_ordering=False):
        """
        :param n_classes: number of classes
        :param n_features: number of predictor variables
        :param samples_before_change: number of steps to run before shuffling classes
        :param callback_on_change: a callable that will be called when classes are shuffled. It should accept this
                                   CardSortingEnv instance as an argument.
        :param show_current_ordering: if True, variables indicating the current ordering will be part of the state space
        """
        self.n_classes = n_classes
        self.n_features = n_features
        self.n_samples = 100
        self.callback_on_change = callback_on_change
        self.show_current_ordering = show_current_ordering

        self.X, self.y = make_classification(
            n_samples=self.n_samples,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_repeated=0,
            n_classes=n_classes,
            class_sep=5,
            n_clusters_per_class=1,
            scale=1/10
        )
        self.X = StandardScaler().fit_transform(self.X)
        self.observation_space = Box(low=self.X.min(), high=self.X.max(), shape=[n_features if not show_current_ordering else n_features + n_classes])
        self.action_space = Discrete(n=n_classes)
        self.i = 0
        self.current_ordering = 0
        self.shuffled_y = self.y.copy()

        self.change_counter = 0
        self.samples_before_change = samples_before_change

    def shuffle_classes(self):
        self.current_ordering = (self.current_ordering + 1) % self.n_classes
        self.shuffled_y = (self.y + self.current_ordering) % self.n_classes

    def _get_state_vector(self):
        state_vec = self.X[self.i]

        if self.show_current_ordering:
            ordering_dummies = np.zeros(self.n_classes)
            ordering_dummies[self.current_ordering] = 1
            state_vec = np.concatenate((state_vec, ordering_dummies))

        return state_vec

    def reset(self):
        self.shuffle_classes()
        self.i = 0
        self.change_counter = 0
        return self._get_state_vector()

    def step(self, action, check_action_only=False):

        reward = 1.0 if action == self.shuffled_y[self.i] else -1.0

        if check_action_only:
            return self._get_state_vector(), reward, False, None

        self.change_counter = (self.change_counter + 1) % self.samples_before_change
        if self.change_counter == 0:
            if callable(self.callback_on_change):
                self.callback_on_change(self)
            self.shuffle_classes()

        self.i = (self.i + 1) % self.n_samples

        return self._get_state_vector(), reward, False, None
