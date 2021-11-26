import numpy as np


class SarsaAgent:
    def __init__(self, env, alpha=None, gamma=None, epsilon=None):
        """ Solves the shortest path problem using SARSA
            :input Maze env           : The maze environment in which we seek to
                                        find the shortest path.
            :input float alpha        : Parameter for step size
            :input float gamma        : The discount factor.
            :input float epsilon      : Probability of random action
        """
        self.n_actions = env.n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        r = env.rewards
        self.Q = np.copy(r)
        self.N = np.zeros((env.n_states, self.n_actions))

    def choose_action(self, state, rng, action_list):
        if np.random.random() < self.epsilon:
            action = rng.choice(action_list)
        else:
            action = np.argmax(self.Q[state, :])
        return action

    def learn(self, state, action, reward, state2, action2):
        self.N[state, action] += 1
        self.Q[state, action] = self.Q[state, action] + 1 / (self.N[state, action] ** self.alpha) * (reward + self.gamma * self.Q[state2, action2] - self.Q[state, action])


