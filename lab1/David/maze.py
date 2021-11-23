import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import json
# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED = '#FFC4CC'
LIGHT_GREEN = '#95FD99'
BLACK = '#000000'
WHITE = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'

# Random number generator object
rng = np.random.default_rng()


class Maze:

    # Actions
    STAY = 0
    MOVE_LEFT = 1
    MOVE_RIGHT = 2
    MOVE_UP = 3
    MOVE_DOWN = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100

    def __init__(self, maze, weights=None, random_rewards=False, stand_still=False, life_mean=None):
        """ Constructor of the environment Maze.
        Input: life_mean        The mean of a geometric distribution giving the life of the player
                                after they have been poisoned.

        Input: stand_still      Boolean which indicates if the minotaur is allowed to stand still
                                or not.
        """
        self.maze = maze
        self.life_mean = life_mean
        self.stand_still = stand_still
        self.actions = self.__actions()
        self.states, self.map = self.__states()
        self.n_actions = len(self.actions)
        self.n_states = len(self.states)
        self.transition_probabilities = self.__transitions()

        self.rewards = self.__rewards(weights=weights,
                                      random_rewards=random_rewards)

    def __actions(self):
        actions = dict()
        actions[self.STAY] = (0, 0)
        actions[self.MOVE_LEFT] = (0, -1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP] = (-1, 0)
        actions[self.MOVE_DOWN] = (1, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        end = False
        s = 0
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i, j] != 1:
                    # The introduction of the minotaur could simply be implemented
                    # by introducing more states (quadratic increase). Each state
                    # represents the location of the player as well as the location
                    # of the minotaur
                    for m in range(self.maze.shape[0]):
                        for n in range(self.maze.shape[1]):
                            states[s] = (i, j, m, n)
                            map[(i, j, m, n)] = s
                            s += 1
        # Absorbing dead state. If the player dies by poisoning they will transition to this state
        if self.life_mean != None:
            states[s] = "Dead"
            map["Dead"] = s
            s += 1
        return states, map

    def __move(self, state, action) -> int:
        # Returns a random next state out of the possible next states for a (state, action) pair, based on transition probabilities
        next_states = np.where(
            self.transition_probabilities[:, state, action] > 0)[0]
        next_state = rng.choice(
            next_states, p=self.transition_probabilities[next_states, state, action])
        return next_state

    """
    Original Move Function:
    """
    # def __move(self, state, action):
    #     """ Makes a step in the maze, given a current position and an action.
    #         If the action STAY or an inadmissible action is used, the agent stays in place.

    #         :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
    #     """
    #     # Compute the future position given current (state, action)
    #     row = self.states[state][0] + self.actions[action][0]
    #     col = self.states[state][1] + self.actions[action][1]
    #     # Is the future position an impossible one ?
    #     hitting_maze_walls = (row == -1) or (row == self.maze.shape[0]) or \
    #         (col == -1) or (col == self.maze.shape[1]) or \
    #         (self.maze[row, col] == 1)
    #     # Based on the impossiblity check return the next state.
    #     if hitting_maze_walls:
    #         return state
    #     else:
    #         return self.map[(row, col)]

    def __possible_transitions(self, state, action) -> list:
        """Description: Method which takes in a state, action pair and returns the possible next states 
        for this pair. Takes into account the rules which govern the movements of both the
        player and the minotaur. 

        Input: state            The current state of the player

        Input: action           The action chosen by the player

        Output: states          A list containing the possible next state(s) given (s,a)
        """

        # If the player has died by poisoning and is in the absorbind "Dead" state, this state is returned
        if (self.states[state] == "Dead"):
            return [state]
        # Compute the future player position given current (state, action)
        player_row = self.states[state][0] + self.actions[action][0]
        player_col = self.states[state][1] + self.actions[action][1]
        # Is the future player position an impossible one ?
        hitting_maze_walls = (player_row == -1) or (player_row == self.maze.shape[0]) or \
            (player_col == -1) or (player_col == self.maze.shape[1]) or \
            (self.maze[player_row, player_col] == 1)
        # Check if player has reached end state
        end_state = (self.maze[self.states[state][0:2]] == 2)
        # Check if player is eaten
        dead_state = (
            self.states[state][0:2] == self.states[state][2:])
        # If next state is impossible or if the player has reached the end state
        # or if the minotaur has eaten the player, return original state
        if hitting_maze_walls or end_state or dead_state:
            return [state]
        else:
            possible_states = []
            for i in range(self.stand_still == False, len(self.actions)):
                minotaur_row = self.states[state][2] + self.actions[i][0]
                minotaur_col = self.states[state][3] + self.actions[i][1]
                outside_of_maze = (minotaur_row == -1) or (minotaur_row == self.maze.shape[0]) or \
                    (minotaur_col == -1) or (minotaur_col ==
                                             self.maze.shape[1])
                if not outside_of_maze:
                    possible_states.append(
                        self.map[(player_row, player_col, minotaur_row, minotaur_col)])
            return possible_states

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states, self.n_states, self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. The randomness comes from the minotaur's movements.
        # self.__move will return a random next state, so for calculating transisiton probabilities
        # we have to use another method. next_s will also be a list of possible next states, given an
        # action
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__possible_transitions(s, a)
                # next_s = self.__move(s, a)
                # If life_mean is included each move will have a possibility of ending up in 'Dead' state.
                # If the player is in dead state they should transition to dead state. If the player is
                # not in "Dead" state they will transition to this state with a probability of
                # 1/life_mean.
                if self.life_mean != None and len(next_s) > 1:
                    prob = ((self.life_mean-1)/(self.life_mean))/len(next_s)
                    transition_probabilities[self.map["Dead"],
                                             s, a] = 1/(self.life_mean)
                    for s_prim in next_s:
                        if s_prim != self.map["Dead"]:
                            transition_probabilities[s_prim, s, a] = prob
                else:
                    prob = 1/len(next_s)
                    for s_prim in next_s:
                        transition_probabilities[s_prim, s, a] = prob

        return transition_probabilities

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions))

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__possible_transitions(s, a)
                    # Reward for being eaten
                    if self.states[s][0:2] == self.states[s][2:]:
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    # Reward for dying of poisoning is the same
                    # as for walking into walls or being eaten
                    elif self.states[s] == "Dead":
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    # Reward for walking into wall
                    elif len(next_s) == 1 and self.maze[self.states[s][0:2]] != 2:
                        rewards[s, a] = self.IMPOSSIBLE_REWARD
                    # Reward for reaching the exit
                    elif self.maze[self.states[s][0:2]] == 2:
                        rewards[s, a] = self.GOAL_REWARD
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s, a] = self.STEP_REWARD

                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[self.states[next_s]] < 0:
                        row, col = self.states[next_s]
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s, a]
                        # With probability 0.5 the reward is
                        r2 = rewards[s, a]
                        # The average reward
                        rewards[s, a] = 0.5*r1 + 0.5*r2
        # If the weights are described by a weight matrix
        else:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s, a)
                    i, j = self.states[next_s]
                    # Simply put the reward as the weights of the next state.
                    rewards[s, a] = weights[i][j]

        return rewards

    def simulate(self, start, policy, method):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods)
            raise NameError(error)

        path = list()
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1]
            # Initialize current state and time
            t = 0
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s, t])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
                s = next_s
        if method == 'ValIter':
            # Initialize current state, next state and time
            t = 1
            s = self.map[start]
            # Add the starting position in the maze to the path
            path.append(start)
            # Move to next state given the policy and the current state
            next_s = self.__move(s, policy[s])
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Loop while state is not the goal state or "Dead" state
            while self.states[s] != "Dead" and self.maze[self.states[s][0:2]] != 2:
                # Update state
                s = next_s
                # Move to next state given the policy and the current state
                next_s = self.__move(s, policy[s])
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t += 1
        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)


def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions
    T = horizon

    # The variables involved in the dynamic programming backwards recursions
    V = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1), dtype=int)
    Q = np.zeros((n_states, n_actions))

    # Initialization
    Q = np.copy(r)
    V[:, T] = np.max(Q, 1)
    policy[:, T] = np.argmax(Q, 1)

    # The dynamic programming bakwards recursion
    for t in range(T-1, -1, -1):
        # Update the value function according to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s, a] = r[s, a] + np.dot(p[:, s, a], V[:, t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:, t] = np.max(Q, 1)
        # The optimal action is the one that maximizes the Q function
        policy[:, t] = np.argmax(Q, 1)
    return V, policy


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p = env.transition_probabilities
    r = env.rewards
    n_states = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V = np.zeros(n_states)
    Q = np.zeros((n_states, n_actions))
    BV = np.zeros(n_states)
    # Iteration counter
    n = 0
    # Tolerance error
    tol = (1 - gamma) * epsilon/gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:, s, a], V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:, s, a], V)
        BV = np.max(Q, 1)
        # Show error
        # print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q, 1)
    # Return the obtained policy
    return V, policy


def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK,
               2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows, cols = maze.shape
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)


def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK,
               2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the maze
    rows, cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j, i]]
                     for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols, rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0, 0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    # Modifications have been made to the following part of this function to accommodate
    # for the possibility of the minotaur and the player standing still, as well as for
    # the player dying of poisoning.
    # Index to keep track of last square that the player was on before dying of poisoning
    last_idx = None

    for i in range(len(path)):
        if path[i] == 'Dead':
            if last_idx == None:
                last_idx = i-1
            grid.get_celld()[(path[last_idx][0:2])
                             ].set_facecolor(LIGHT_RED)
            grid.get_celld()[(path[last_idx][0:2])].get_text().set_text(
                'Player is dead')
        else:
            grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_ORANGE)
            grid.get_celld()[(path[i][0:2])].get_text().set_text('Player')
            grid.get_celld()[(path[i][2:])].set_facecolor(LIGHT_PURPLE)
            grid.get_celld()[(path[i][2:])].get_text().set_text('Minotaur')
        if i > 0:
            if path[i] == path[i-1]:
                if path[i][0:2] == path[i][2:]:
                    grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_RED)
                    grid.get_celld()[(path[i][0:2])].get_text().set_text(
                        'Player is eaten')
                elif path[i][0:2] == (6, 5):
                    grid.get_celld()[(path[i][0:2])].set_facecolor(LIGHT_GREEN)
                    grid.get_celld()[(path[i][0:2])].get_text().set_text(
                        'Player is out')
            elif path[i] != "Dead":
                if path[i-1][0:2] != path[i][2:] and path[i-1][0:2] != path[i][0:2]:
                    grid.get_celld()[(path[i-1][0:2])
                                     ].set_facecolor(col_map[maze[path[i-1][0:2]]])
                    grid.get_celld()[(path[i-1][0:2])].get_text().set_text('')
                if path[i-1][2:] != path[i][0:2] and path[i-1][2:] != path[i][2:]:
                    grid.get_celld()[(path[i-1][2:])
                                     ].set_facecolor(col_map[maze[path[i-1][2:]]])
                    grid.get_celld()[(path[i-1][2:])].get_text().set_text('')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(1)


def policy_evaluation(env, policy, horizon):
    """Evaluates a policy based on the probability of reaching the end state given a time horizon.

    :Input env          : The maze environment for which the policy was calculated

    :Input policy       : The calculated policy for the previous maze environment

    :Input horizon      : The time horizon for which the policy should be evaluated

    :Output Value       : The value function for all states of the environment under the provided policy
    """
    p = np.copy(env.transition_probabilities)
    n_states = env.n_states
    n_actions = env.n_actions
    T = horizon

    r = np.zeros((n_states, n_actions), dtype=int)
    # Initiate rewards to be 1 if state is end state and player and minotaur are not in the same
    # cell, and zero otherwise
    for s in range(n_states):
        for a in range(n_actions):
            if env.states[s][0:2] == (6, 5) and env.states[s][0:2] != env.states[s][2:]:
                r[s, a] = 1

    # Initialization of the value function for the backwards recursion
    V = np.zeros((n_states, T+1))
    # The value of a state at the final time is equal to the instantaneous reward for that state, in
    # accordance with Bellman's equations
    a_p = policy[:, T]
    V[:, T] = r[range(n_states), a_p]

    # Compute backwards recursion, following the given policy
    for t in range(T-1, -1, -1):
        for s in range(n_states):
            V[s, t] = np.dot(p[:, s, policy[s, t]], V[:, t+1])
    start_state = env.map[(0, 0, 6, 5)]
    return V[start_state, 0]
