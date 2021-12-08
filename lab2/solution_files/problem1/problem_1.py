import numpy as np
import gym
from tqdm import trange
from DQN_agent import DQNAgent


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_episodes = 200                             # Number of episodes
discount_factor = 0.99                        # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

max_size = 10000                             # Maximum length of replay buffer
buffer_init = 0.2                            # Percentage of buffer which is
# filled before learning starts
train_batch = 64                             # Amount of experiences in a batch
# Update frequency of target network
update_freq = int(1*max_size/train_batch)
lr = 1e-3                                    # Learning rate
clip_val = 1                                 # Gradient clipping value
epsilon_max = 0.9                            # Exploration parameter max,min
epsilon_min = 0.1
start_annealing = 0*N_episodes             # Percentage of episodes when
stop_annealing = 0.9*N_episodes              # learning starts and ends
exponential = True                           # Exponential decay, else linear
neurons = 64                                 # Number of neurons in hidden
# layers of neural nets


# We will use these variables to compute the average episodic reward and
# the average number of steps per episode

episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode
episode_loss_list = []
loss_list = []
eps_list = []
updates = 0


# Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
#agent = RandomAgent(n_actions)
agent = DQNAgent(neurons, n_actions, dim_state, lr,
                 discount_factor, train_batch, clip_val, max_size=max_size)
agent.initialize_buffer(buffer_init, env)

EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
steps = 0

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state = env.reset()
    total_episode_reward = 0.
    total_episode_loss = 0.
    t = 0
    while not done:

        if exponential:
            # Calculate epsilon value, exponential annealing
            if i >= round(start_annealing):
                epsilon = max(epsilon_min, epsilon_max*(epsilon_min/epsilon_max)
                              ** ((i-start_annealing)/(stop_annealing-start_annealing-1)))
            else:
                epsilon = epsilon_max

        else:
            # Calculate epsilon value, linear annealing
            if i >= round(start_annealing):
                epsilon = max(epsilon_min, epsilon_max - (epsilon_max-epsilon_min)
                              * (i-start_annealing)/(stop_annealing-start_annealing-1))
            else:
                epsilon = epsilon_max

        # Take a random action
        action = agent.forward(state, epsilon)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        # Append experience to buffer
        experience = {'state': state, 'action': action,
                      'reward': reward, 'next_state': next_state, 'done': done}
        agent.buffer.append(experience)

        # Update episode reward
        total_episode_reward += reward

        # Perform learning step for the agent
        loss = agent.learn()
        total_episode_loss += loss
        loss_list.append(loss)

        # Update target if enough steps have passed
        if steps % update_freq == 0:
            agent.update_target()
            updates += 1

        # Update state for next iteration
        state = next_state
        t += 1
        steps += 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)
    eps_list.append(epsilon)
    episode_loss_list.append(total_episode_loss)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))

