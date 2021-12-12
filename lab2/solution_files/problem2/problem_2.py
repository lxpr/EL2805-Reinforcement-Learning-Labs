import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import DDPGAgent, RandomAgent


# def plot_q_values(network, num_step=100):
#     y = np.linspace(0, 1.5, num_step)
#     w = np.linspace(-np.pi, np.pi, num_step)
#     Y, W = np.meshgrid(y, w)
#     states = np.zeros((num_step ** 2, 8))
#
#     states[:, 1] = Y.flatten()
#     states[:, 4] = W.flatten()
#
#     values = network(torch.tensor(states, dtype=torch.float32, requires_grad=False))
#     max_values, argmax_values = values.max(1)
#
#     max_Q = max_values.detach().numpy().reshape((num_step, num_step))
#     argmax_Q = argmax_values.detach().numpy().reshape((num_step, num_step))
#
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.plot_surface(Y, W, max_Q, cmap='viridis', edgecolor='none')
#
#     ax.set_title('Maximum Q values')
#     ax.set_xlabel('y')
#     ax.set_ylabel('$\omega$')
#     ax.set_zlabel('Max Q')
#
#     fig = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.plot_surface(Y, W, argmax_Q, cmap='viridis', edgecolor='none')
#
#     ax.set_title('Optimal Q actions')
#     ax.set_xlabel('y')
#     ax.set_ylabel('$\omega$')
#     ax.set_zlabel('Action')
#     ax.set_zticks([0, 1, 2, 3])
#     ax.set_zticklabels(['Stay', 'Left E.', 'Main E.', 'Right E.'])
#     plt.show()


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
N_episodes = 500                             # Number of episodes
discount_factor = 0.99                       # Value of the discount factor
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
agent = DDPGAgent(neurons, n_actions, dim_state, lr,
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
        loss = agent.learn(combined=True)
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

# Save the network
# torch.save(agent.network, 'neural-network-1.pth')
# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)

plt.figure()
plt.plot(eps_list)
plt.title("Epsilon value over episodes")
plt.ylabel("Epsilon")
plt.xlabel("Episodes")
plt.grid(alpha=0.3)
plt.show()
