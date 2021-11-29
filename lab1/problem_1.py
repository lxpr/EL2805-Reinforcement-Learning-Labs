import maze as mz
import maze_keys as mz_k
import numpy as np
from IPython import display
from matplotlib import pyplot as plt
from QLearning import qlearning


def exit_probability_fun(env_1):
    exit_probability = np.empty(30)
    for horizon in range(1, 31):
        print(horizon)
        V, policy = mz.dynamic_programming(env_1, horizon)
        exit_probability[horizon -
                         1] = mz.policy_evaluation(env_1, policy, horizon)
    return exit_probability


# Run this code to see solution to problem 1b
def problem_1b(env, maze, minotaur_position=(6, 5)):
    # Finite horizon
    horizon = 20
    # Solve the MDP problem with dynamic programming
    V, policy = mz.dynamic_programming(env, horizon)
    mz.illustrate_policy(env, maze, policy, minotaur_position)


# Run this code to see solution to problem 1c
def problem_1c(env, maze):
    exit_probability = exit_probability_fun(env)
    plt.figure()
    plt.plot(np.arange(1, 31), exit_probability, label="Exit Probability")
    plt.plot(np.ones(31), linestyle="--", color="k", label="$P=1$")
    plt.legend()
    plt.title("Exit Probability When the Minotaur Must Move")
    plt.ylabel("P")
    plt.xlabel("Horizon")
    plt.xlim([0, 30])
    plt.xticks([1, 4, 9, 14, 19, 24, 29], labels=[1, 5, 10, 15, 20, 25, 30])

    env = mz.Maze(maze, stand_still=True)

    exit_probability = exit_probability_fun(env)
    plt.figure()
    plt.plot(np.arange(1, 31), exit_probability, label="Exit Probability")
    plt.plot(np.ones(31), linestyle="--", color="k", label="$P=1$")
    plt.legend()
    plt.title("Exit Probability When the Minotaur Can Stand Still")
    plt.ylabel("P")
    plt.xlabel("Horizon")
    plt.xlim([0, 30])
    plt.xticks([1, 4, 9, 14, 19, 24, 29], labels=[1, 5, 10, 15, 20, 25, 30])

    plt.show()


def problem_1e(maze):

    env = mz.Maze(maze, stand_still=False, life_mean=30)

    print("The Minotaur Can't Stand Still")
    nr_escape = 0
    start = (0, 0, 6, 5)
    method = 'ValIter'
    V, policy = mz.value_iteration(env, 1/3, epsilon=0.0001)
    N = 10000
    for i in range(N):
        path_4 = env.simulate(start, policy, method)
        if path_4[-1] != "Dead":
            if maze[path_4[-1][0:2]] == 2 and path_4[-1][0:2] != path_4[-1][2:]:
                nr_escape += 1

    print("Estimated probability of successful escape:")
    print(nr_escape/N)

    env = mz.Maze(maze, stand_still=True, life_mean=30)
    print("The Minotaur Can Stand Still")
    nr_escape = 0
    start = (0, 0, 6, 5)
    method = 'ValIter'
    V, policy = mz.value_iteration(env, 1/3, epsilon=0.0001)
    N = 10000
    for i in range(N):
        path_4 = env.simulate(start, policy, method)
        if path_4[-1] != "Dead":
            if maze[path_4[-1][0:2]] == 2 and path_4[-1][0:2] != path_4[-1][2:]:
                nr_escape += 1

    print("Estimated probability of successful escape:")
    print(nr_escape/N)


def problem_1g():
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 3],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]
    ])

    env = mz_k.Maze(maze, life_mean=50, prob_random=0.65)
    V, policy = mz_k.value_iteration(env, 0.9, epsilon=0.01)
    nr_escape = 0
    method = 'ValIter'
    start = (0, 0, 6, 5, 0)
    N = 10000

    for i in range(N):
        path_7 = env.simulate(start, policy, method)
        if path_7[-1] != "Dead":
            if maze[path_7[-1][0:2]] == 2 and path_7[-1][0:2] != path_7[-1][2:] and path_7[-1][-1] == 1:
                nr_escape += 1
    print("Estimated probability of successful escape:")
    print(nr_escape/N)


def problem_1h():
    maze_Q = np.array([
        [0, 0, 1, 0, 0, 0, 0, 3],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]
    ])

    env_Q = mz_k.Maze(maze_Q, life_mean=50, prob_random=0.65)
    epsilon = 0.1
    print("Beginning Q-Learning for epsilon={}:\n".format(epsilon))
    Q, policy_Q, V_convergence = qlearning(env_Q, epsilon, gamma=0.9, N=50000)

    plt.figure()
    plt.title("Convergence of Starting State for epsilon={}".format(epsilon))
    plt.plot(V_convergence)

    epsilon = 0.5
    print("Beginning Q-Learning for epsilon={}:\n".format(epsilon))
    Q, policy_Q, V_convergence = qlearning(env_Q, epsilon, gamma=0.9, N=50000)

    plt.figure()
    plt.title("Convergence of Starting State for epsilon={}".format(epsilon))
    plt.plot(V_convergence)

    epsilon = 0.1
    alpha = 0.6
    print(
        "Beginning Q-Learning for epsilon={0} and alpha={1}:\n".format(epsilon, alpha))
    Q, policy_Q, V_convergence = qlearning(
        env_Q, epsilon, alpha=0.6, gamma=0.9, N=50000)

    plt.figure()
    plt.title("Convergence of Starting State for epsilon={0} and alpha={1}".format(
        epsilon, alpha))
    plt.plot(V_convergence)

    epsilon = 0.5
    alpha = 1
    print("Beginning Q-Learning for epsilon={} and alpha={}:\n".format(epsilon, alpha))
    Q, policy_Q, V_convergence = qlearning(
        env_Q, epsilon, alpha=1, gamma=0.9, N=50000)

    plt.figure()
    plt.title("Convergence of Starting State for epsilon={0} and alpha={1}".format(
        epsilon, alpha))
    plt.plot(V_convergence)

    plt.show()


if __name__ == "__main__":
    maze = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 1, 2, 0, 0]
    ])

    env = mz.Maze(maze)

    # Uncomment the problem that you would like to run code for. Problem_1c takes a while to run, since it runs dynamic programming
    # for a total of 60 different time horizons. The Q-Learning code also takes a while, as it learns and plots
    # convergence for a policy for several values and alpha (exponent of step size)

    # problem_1b(env, maze)
    # problem_1c(env, maze)
    # problem_1e(maze)
    # problem_1g()
    problem_1h()
