import numpy as np
from maze_keys import Maze


def qlearning(env, epsilon, alpha=None, N=50000, gamma=1, random_start=False):
    """Description: Function which implements the Q-Learning algorithm

    Input env:          Environment for which the algorithm should learn the optimal policy


    Input epsilon:      Parameter which tells the Q-Learning algorithm's behaviour policy
                        with which probability to act greedy, as well as to explore

    Input alpha:        Step size exponent, if set to a different value than 2/3


    Input N:            Number of iterations of the algorithm


    Output Q, policy:    Returns step-action value function Q for the learned policy, as well as the policy itself
    """

    # Random number generator object
    rng = np.random.default_rng()

    # Initialization
    P = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions

    # Q function
    Q = np.zeros((n_states, n_actions))
    # Number of times a specific state-action pair has been observed
    n_sa = np.zeros(Q.shape)
    V_convergence = np.empty(N)
    for n in range(N):
        if n % 1000 == 0:
            print(n)
        if random_start:
            # Initiate state to random state
            s = rng.choice(n_states)
            start = True
        else:
            # Initiate to starting state
            s = env.map[(0, 0, 6, 5, 0)]
            start = False
        while start or (env.states[s] != "Dead" and not (env.maze[env.states[s][0:2]] == 2 and env.states[s][-1] == 1) and not(env.states[s][0:2] == env.states[s][2:4])):
            # Generate an action using the epsilon-soft behaviour policy
            start = False
            if rng.choice([0, 1], p=[1-epsilon, epsilon]):
                # With probability epsilon choose a random action
                a = rng.choice(n_actions)
            else:
                # With probability 1-epsilon choose epsilon greedy wrt Q. If several actions have the same value, choose randomly between them
                a = rng.choice(np.where(Q[s, :] == max(Q[s, :]))[0])

            # Collect reward based on state and action
            r_n = r[s, a]

            # Generate a new state based on the transition probabilities given by the state and action
            next_states = np.where(P[:, s, a] > 0)[0]
            probs = P[next_states, s, a]
            next_s = rng.choice(next_states, p=P[next_states, s, a])

            # Update n_sa
            n_sa[s, a] += 1

            step_size = 1/(n_sa[s, a]**(2/3)
                           ) if alpha == None else 1/(n_sa[s, a]**alpha)

            # Policy improvement
            Q[s, a] += step_size*(r_n + gamma*max(Q[next_s, :]) - Q[s, a])

            # Move on to next state
            s = next_s
        V_convergence[n] = max(Q[env.map[(0, 0, 6, 5, 0)], :])
    # The policy returned by the function is the greedy policy wrt Q
    policy = np.argmax(Q, 1)
    print("Nr of times initial state has been visited")
    print(n_sa[env.map[(0, 0, 6, 5, 0)], :])
    return Q, policy, V_convergence


if __name__ == "__main__":
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 3],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]
    ])

    env = Maze(maze, life_mean=50)
    epsilon = 0.5

    # print(env.states[2066])
    # next_states = np.where(env.transition_probabilities[:, 2066, :] > 0)[0]
    # print(next_states)
    Q, policy = qlearning(env, epsilon, N=20)
