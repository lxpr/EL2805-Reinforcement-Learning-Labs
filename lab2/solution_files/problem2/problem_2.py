import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_soft_updates import soft_updates
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


# Import and initialize the discrete Lunar Lander Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
N_episodes = 300                             # Number of episodes
discount_factor = 0.99                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
m = len(env.action_space.high)               # dimensionality of the action
dim_state = len(env.observation_space.high)  # State dimensionality

max_size = 30000                             # Maximum length of replay buffer
buffer_init = 1                           # Percentage of buffer which is
# filled before learning starts
train_batch = 64                             # Amount of experiences in a batch
# Update frequency of actor and target network
update_freq = 2
lr_actor = 5 * pow(10, -5)                   # Learning rate
lr_critic = 5 * pow(10, -4)                  # Learning rate
clip_val = 1                                 # Gradient clipping value
neurons_1 = 400                                 # Number of neurons in hidden layer 1
neurons_2 = 200                                 # Number of neurons in hidden layer 2
tau = 1e-3                                      # Soft update parameter
# layers of neural nets


# We will use these variables to compute the average episodic reward and
# the average number of steps per episode

episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode
episode_loss_list = []
# episode_policy_loss_list = []
loss_list = []
# policy_loss_list = []
updates = 0


# Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
#agent = RandomAgent(n_actions)
agent = DDPGAgent(neurons_1, neurons_2, m, dim_state, lr_actor, lr_critic,
                 discount_factor, train_batch, clip_val, max_size=max_size)
agent.initialize_buffer(buffer_init, env)

EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
steps = 0

for i in EPISODES:
    # Reset environment data and initialize variables
    done = False
    state = env.reset()
    total_episode_reward = 0.
    total_episode_loss = 0.
    total_episode_policy_loss = 0.
    t = 0
    agent.n_t = np.zeros(agent.m)
    while not done:

        # Take a random action
        action = agent.forward(state)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)

        # Append experience to buffer
        exp = (state, action, reward, next_state, done)
        agent.buffer.append(exp)

        # Update episode reward
        total_episode_reward += reward

        # Perform learning step for the agent
        # Update actor and target if enough steps have passed
        if steps % update_freq == 0:
            loss, policy_loss = agent.learn(combined=False, actor_update=True)
            agent.actor_target = soft_updates(agent.actor, agent.actor_target, tau)
            agent.critic_target = soft_updates(agent.critic, agent.critic_target, tau)

            updates += 1
        else:
            loss, policy_loss = agent.learn(combined=False, actor_update=False)
        total_episode_loss += loss
        # print('loss:', loss)
        # print('policy_loss:', policy_loss)
        # total_episode_policy_loss += policy_loss
        loss_list.append(loss)
        # policy_loss_list.append(policy_loss)

        # Update state for next iteration
        state = next_state
        t += 1
        steps += 1
        agent.noise()

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)
    episode_loss_list.append(total_episode_loss)
    # episode_policy_loss_list.append(total_episode_policy_loss)

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
torch.save(agent.actor, 'neural-network-2-actor.pth')
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
plt.show()
