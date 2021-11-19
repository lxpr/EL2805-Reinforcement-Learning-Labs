import maze as mz
import numpy as np
from IPython import display

# Modified maze from lab0. Now corresponding to maze in lab1

maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])

env = mz.Maze(maze, stand_still=True)
mz.dynamic_programming(env, 30)


# Finite horizon
horizon = 30
# Solve the MDP problem with dynamic programming
V, policy = mz.dynamic_programming(env, horizon)

# Simulate the shortest path starting from position A
method = 'DynProg'
start = (0, 0, 6, 5)
path = env.simulate(start, policy, method)

mz.policy_evaluation(env, policy, horizon)
