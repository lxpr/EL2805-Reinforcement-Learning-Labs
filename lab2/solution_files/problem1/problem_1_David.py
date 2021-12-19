import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DQN_agent import DQNAgent, RandomAgent, NeuralNet
from collections import namedtuple
import os.path
import json
from DQN_check_solution import check_solution
from json.decoder import JSONDecodeError

Experience = namedtuple(
    'Experience', 'state, action, next_state, reward, done')


def plot_q_values(network, num_step=100, plot=True):
    y = np.linspace(0, 1.5, num_step)
    w = np.linspace(-np.pi, np.pi, num_step)
    Y, W = np.meshgrid(y, w)
    states = np.zeros((num_step ** 2, 8))

    states[:, 1] = Y.flatten()
    states[:, 4] = W.flatten()

    values = network(torch.tensor(
        states, dtype=torch.float32, requires_grad=False))
    max_values, argmax_values = values.max(1)

    max_Q = max_values.detach().numpy().reshape((num_step, num_step))
    argmax_Q = argmax_values.detach().numpy().reshape((num_step, num_step))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(Y, W, max_Q, cmap='viridis', edgecolor='none')

    ax.set_title('Maximum Q values')
    ax.set_xlabel('y')
    ax.set_ylabel('$\omega$')
    ax.set_zlabel('Max Q')

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(Y, W, argmax_Q, cmap='viridis', edgecolor='none')

    ax.set_title('Optimal Q actions')
    ax.set_xlabel('y')
    ax.set_ylabel('$\omega$')
    ax.set_zlabel('Action')
    ax.set_zticks([0, 1, 2, 3])
    ax.set_zticklabels(['Stay', 'Left E.', 'Main E.', 'Right E.'])

    if plot:
        plt.show()


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
N_episodes = 5                            # Number of episodes
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
epsilon_min = 0.05
start_annealing = int(0*N_episodes)             # Percentage of episodes when
stop_annealing = int(0.8*N_episodes)             # learning starts and ends
exponential = True                           # Exponential decay, else linear
neurons = 64                                 # Number of neurons in hidden
# layers of neural nets

# Dictionary to save parameters for network
parameter_dict = {"N_episodes": N_episodes, "discount_factor": discount_factor,
                  "n_ep_running_average": n_ep_running_average, "n_actions": n_actions,
                  "dim_state": dim_state, "max_size": max_size, "buffer_init": buffer_init,
                  "train_batch": train_batch, "update_freq": update_freq, "lr": lr, "clip_val": clip_val,
                  "epsilon_max": epsilon_max, "epsilon_min": epsilon_min, "start_annealing": start_annealing,
                  "stop_annealing": stop_annealing, "exponential": exponential, "neurons": neurons}


def training_process(name, combined, plot=False, early_stop=True):
    # Training process

    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode

    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode
    episode_loss_list = []
    loss_list = []
    eps_list = []
    updates = 0


# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
# agent = RandomAgent(n_actions)
    agent = DQNAgent(neurons, n_actions, dim_state, lr,
                     discount_factor, train_batch, clip_val, max_size=max_size)
    agent.initialize_buffer(buffer_init, env)

    EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
    steps = 0
    stop_training = False

    for i in EPISODES:
        # Reset enviroment data and initialize variables
        if stop_training:
            break
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
            experience = Experience(state, action, next_state, reward, done)
            agent.buffer.append(experience)

            # Update episode reward
            total_episode_reward += reward

            # Perform learning step for the agent
            loss = agent.learn(combined=combined)
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
        avg_last_50 = running_average(
            episode_reward_list, n_ep_running_average)[-1]
        EPISODES.set_description(
            "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
                i, total_episode_reward, t,
                avg_last_50,
                running_average(episode_number_of_steps, n_ep_running_average)[-1]))

        if early_stop and avg_last_50 > 220:
            stop_training = True
            plot_episodes = i
            stopped_at = i + 1

    if not stop_training:
        plot_episodes = N_episodes-1
        stopped_at = N_episodes
    # Save the network
    torch.save(agent.network, f'Networks/{name}.pth')
    # Plot Rewards and steps
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
    ax[0].plot([i for i in range(1, plot_episodes+2)],
               episode_reward_list, label='Episode reward')
    ax[0].plot([i for i in range(1, plot_episodes+2)], running_average(
        episode_reward_list, n_ep_running_average), label='Avg. episode reward')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Total reward')
    ax[0].set_title('Total Reward vs Episodes')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot([i for i in range(1, plot_episodes+2)],
               episode_number_of_steps, label='Steps per episode')
    ax[1].plot([i for i in range(1, plot_episodes+2)], running_average(
        episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Total number of steps')
    ax[1].set_title('Total number of steps vs Episodes')
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.savefig(
        f'Networks/Plots/{name}_rewards_steps.png', bbox_inches='tight')

    plt.figure()
    plt.plot(eps_list)
    plt.title("Epsilon value over episodes")
    plt.ylabel("Epsilon")
    plt.xlabel("Episodes")
    plt.grid(alpha=0.3)
    plt.savefig(f'Networks/Plots/{name}_epsilon.png', bbox_inches='tight')

    if plot:
        plt.show()
    return stopped_at


def simulate_random(N_EPISODES):
    # Reward
    episode_reward_list = []  # Used to store episodes reward
    agent = RandomAgent(n_actions)
    # Simulate episodes
    EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
    for i in EPISODES:
        EPISODES.set_description("Episode {}".format(i))
        # Reset enviroment data
        done = False
        state = env.reset()
        total_episode_reward = 0.
        while not done:
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            action = agent.forward(state)
            next_state, reward, done, _ = env.step(action)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state

        # Append episode reward
        episode_reward_list.append(total_episode_reward)

        # Close environment
        env.close()
    return episode_reward_list


def simulate_DQN(agent, N_EPISODES):

    env.reset()
    episode_reward_list = []  # Used to store episodes reward
    EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
    for i in EPISODES:
        EPISODES.set_description("Episode {}".format(i))
        # Reset enviroment data
        done = False
        state = env.reset()
        total_episode_reward = 0.
        while not done:
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            q_values = agent(torch.tensor([state]))
            _, action = torch.max(q_values, axis=1)
            next_state, reward, done, _ = env.step(action.item())

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state

        # Append episode reward
        episode_reward_list.append(total_episode_reward)

        # Close environment
        env.close()

    return episode_reward_list


def compare(number, plot=False):
    """
    Number refers to network number to compare to the random agent
    """
    # Compare Q-network with random agent
    env.reset()
    random_agent = RandomAgent(n_actions)

    # Parameters
    N_EPISODES = 50            # Number of episodes to run for trainings

    episode_reward_list = simulate_random(N_EPISODES)

    avg_reward = np.mean(episode_reward_list)
    confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)

    print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
        avg_reward,
        confidence))

    # Plot for random agent
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 9))
    ax.plot([i for i in range(1, N_EPISODES+1)], running_average(
        episode_reward_list, n_ep_running_average), label='Random Agent')

    # Compare to DQN network
    agent = torch.load(f'Networks/neural-network-{number}.pth')

    episode_reward_list = simulate_DQN(agent, N_EPISODES)

    avg_reward = np.mean(episode_reward_list)
    confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)
    print('Policy achieves an average total reward of {:.1f} +/- {:.1f} with confidence 95%.'.format(
        avg_reward,
        confidence))

    ax.plot([i for i in range(1, N_EPISODES+1)], running_average(
        episode_reward_list, n_ep_running_average), label='DQN Agent')

    ax.set_xlabel('Episodes')
    ax.set_ylabel('Total reward')
    ax.set_title('Total Reward Comparison')

    plt.savefig(
        f'Networks/Plots/comparison.png', bbox_inches='tight')

    ax.legend()
    ax.grid(alpha=0.3)
    if plot:
        plt.show()


if __name__ == "__main__":

    """
    Mode can be "Training, Compare or Q-values
    """

    modes = ("Training", "Compare", "Q-Values", "Test")
    mode = 1
    # Set if you want the training to stop when policy reaches above 220
    early_stop = False
    combined = False
    plot = True
    # Set network number for policy illustration, policy test or comparison
    network_num = 1

    if modes[mode] == "Q-Values":
        filename = f"neural-network-{network_num}"
        filepath = f"Networks/{filename}.pth"
        model = torch.load(filepath)
        plot_q_values(model)
    elif modes[mode] == "Training":
        network_num = 1
        # Set name of model file to be saved
        filename = f"neural-network-{network_num}"
        filepath = f"Networks/{filename}.pth"
        while os.path.isfile(filepath):
            network_num += 1
            filename = f"neural-network-{network_num}"
            filepath = f"Networks/{filename}.pth"
        stopped_at = training_process(
            filename, combined, early_stop=early_stop)
        parameter_dict["early_stop"] = early_stop
        parameter_dict["stopped_at"] = stopped_at
        result, avg_reward, confidence = check_solution(network_num)
        parameter_dict["Combined"] = combined
        parameter_dict["Results"] = {"Result": result,
                                     "avg_reward": avg_reward, "confidence": confidence}

        try:
            with open("Networks/network_parameters.txt", 'r') as f:
                network_dict = json.load(f)
                network_dict[filename] = parameter_dict
            with open("Networks/network_parameters.txt", 'w') as f:
                json.dump(network_dict, f, indent=4)
        except (FileNotFoundError, JSONDecodeError):
            network_dict = {filename: parameter_dict}
            with open("Networks/network_parameters_temp.txt", 'w') as f:
                json.dump(network_dict, f, indent=4)
            print("Error in saved dictionary. Wrote to new called '_temp'!")

    elif modes[mode] == "Compare":
        compare(network_num, plot)

    elif modes[mode] == "Test":
        result, avg_reward, confidence = check_solution(network_num)
