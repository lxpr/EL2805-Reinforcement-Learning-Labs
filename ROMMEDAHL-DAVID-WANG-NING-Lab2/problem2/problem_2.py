

#################################################################################################################
######################################### Look here! ############################################################
#################################################################################################################

# This is the main file used to train, evaluate and compare networks using DDPG.
# If you scroll down to the end of this document, you will see the code which controls what this file does.
# If it is run as is, it will test the final network. It can also be used to train new networks and store the results
# in the Networks folder, together with their plots and parameter dictionary.

# The final network, which is called 'neural-network-2-actor.pth', is an instance of the ActorNet class. It is
# a copy of the network called 'actor-network-1.pth', which is located in the Networks folder.
# DDPG_check_solution_david.py is a version of DDPG_check_solution.py which makes the functionality into a function,
# to export the results and include in the networks_parameters.txt dictionary in the Networks folder.

import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
from DDPG_agent import DDPGAgent, RandomAgent, ActorNeuralNet, CriticNeuralNet, ActorNet, CriticNet
from DDPG_soft_updates import soft_updates
from collections import namedtuple
import os.path
import json
from DDPG_check_solution_david import check_solution
from json.decoder import JSONDecodeError

Experience = namedtuple(
    'Experience', 'state, action, next_state, reward, done')


def plot_q_values(critic_network, actor_network, num_step=100, plot=True):
    y = np.linspace(0, 1.5, num_step)
    w = np.linspace(-np.pi, np.pi, num_step)
    speed = -0*np.ones(num_step**2)
    Y, W = np.meshgrid(y, w)
    states = np.zeros((num_step ** 2, 8))

    states[:, 1] = Y.flatten()
    states[:, 4] = W.flatten()
    states[:, 3] = speed
    states = torch.tensor(
        states, dtype=torch.float32, requires_grad=False)
    actions = actor_network(torch.tensor(
        states, dtype=torch.float32, requires_grad=False))
    engine_actions = actions[:, 1].detach(
    ).numpy().reshape((num_step, num_step))
    values = critic_network(states, actions).detach(
    ).numpy().reshape((num_step, num_step))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(Y, W, values,
                    cmap='viridis', edgecolor='none')

    ax.set_title('Q values')
    ax.set_xlabel('y')
    ax.set_ylabel('$\omega$')
    ax.set_zlabel('Q')

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(Y, W, engine_actions, cmap='viridis', edgecolor='none')

    ax.set_title('Optimal actions')
    ax.set_xlabel('y')
    ax.set_ylabel('$\omega$')
    ax.set_zlabel('Action')
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])

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


# Import and initialize the continuous Lunar Laner Environment
env = gym.make('LunarLanderContinuous-v2')
env.reset()

# Parameters
N_episodes = 300                             # Number of episodes
discount_factor = 0.99                       # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
m = len(env.action_space.high)  # dimensionality of the action
dim_state = len(env.observation_space.high)  # State dimensionality

max_size = 30000                           # Maximum length of replay buffer
buffer_init = 1                           # Percentage of buffer which is
# filled before learning starts
train_batch = 64                             # Amount of experiences in a batch
# Update frequency of actor and target network
update_freq = 2
lr_actor = 5e-5                                    # Learning rate
lr_critic = 5e-4                                    # Learning rate
clip_val = 1                                 # Gradient clipping value
exponential = True                           # Exponential decay, else linear
# Number of neurons in hidden layer 1
neurons_1 = 400
# Number of neurons in hidden layer 2
neurons_2 = 200
tau = 1e-3                                      # Soft update parameter

# Dictionary to save parameters for network
parameter_dict = {"N_episodes": N_episodes, "discount_factor": discount_factor,
                  "n_ep_running_average": n_ep_running_average, "m": m,
                  "dim_state": dim_state, "max_size": max_size, "buffer_init": buffer_init,
                  "train_batch": train_batch, "update_freq": update_freq, "lr_actor": lr_actor,
                  "lr_critic": lr_critic, "clip_val": clip_val, "neurons_1": neurons_1, "neurons_2": neurons_2,
                  "tau": tau}


def training_process(names, combined, plot=False, early_stop=True):
    # Training process

    # We will use these variables to compute the average episodic reward and
    # the average number of steps per episode

    episode_reward_list = []       # this list contains the total reward per episode
    episode_number_of_steps = []   # this list contains the number of steps per episode
    episode_loss_list = []
    loss_list = []
    updates = 0

    actor_name, critic_name = names

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
# agent = RandomAgent(n_actions)
    agent = DDPGAgent(neurons_1, neurons_2, m, dim_state, lr_actor, lr_critic,
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
                loss, policy_loss = agent.learn(
                    combined=False, actor_update=True)
                agent.actor_target = soft_updates(
                    agent.actor, agent.actor_target, tau)
                agent.critic_target = soft_updates(
                    agent.critic, agent.critic_target, tau)

                updates += 1
            else:
                loss, policy_loss = agent.learn()
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
    # Save the networks
    torch.save(agent.actor, f'Networks/{actor_name}.pth')
    torch.save(agent.critic, f'Networks/{critic_name}.pth')

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
        f'Networks/Plots/{actor_name}_rewards_steps.png', bbox_inches='tight')

    if plot:
        plt.show()
    return stopped_at


def simulate_random(N_EPISODES):
    # Reward
    episode_reward_list = []  # Used to store episodes reward
    agent = RandomAgent(m)
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


def simulate_DDPG(agent, N_EPISODES):

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
            action = agent(torch.tensor(state))
            next_state, reward, done, _ = env.step(action.detach().numpy())

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
    # Compare DDPG-network with random agent
    env.reset()

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
    agent = torch.load(f'Networks/actor-network-{number}.pth')

    episode_reward_list = simulate_DDPG(agent, N_EPISODES)

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

    modes = ("Training", "Compare", "Q-Values", "Test", "multi")
    mode = 3
    # Set if you want the training to stop when policy reaches above 220
    early_stop = False
    combined = False
    plot = False
    # Set network number for policy illustration, policy test or comparison
    network_num = 1

    if modes[mode] == "Q-Values":
        actor_filename = f"actor-network-{network_num}"
        actor_filepath = f"Networks/{actor_filename}.pth"
        actor_network = torch.load(actor_filepath)
        critic_filename = f"critic-network-{network_num}"
        critic_filepath = f"Networks/{critic_filename}.pth"
        critic_network = torch.load(critic_filepath)
        plot_q_values(critic_network, actor_network)

    elif modes[mode] == "Training":
        network_num = 1
        # Set name of model file to be saved
        actor_filename = f"actor-network-{network_num}"
        actor_filepath = f"Networks/{actor_filename}.pth"
        critic_filename = f"critic-network-{network_num}"
        critic_filepath = f"Networks/{critic_filename}.pth"
        while os.path.isfile(actor_filepath):
            network_num += 1
            actor_filename = f"actor-network-{network_num}"
            actor_filepath = f"Networks/{actor_filename}.pth"
            critic_filename = f"critic-network-{network_num}"
            critic_filepath = f"Networks/{critic_filename}.pth"
        filenames = (actor_filename, critic_filename)
        stopped_at = training_process(
            filenames, combined, early_stop=early_stop)
        parameter_dict["early_stop"] = early_stop
        parameter_dict["stopped_at"] = stopped_at
        result, avg_reward, confidence = check_solution(network_num)
        parameter_dict["Combined"] = combined
        parameter_dict["Results"] = {"Result": result,
                                     "avg_reward": avg_reward, "confidence": confidence}

        try:
            with open("Networks/network_parameters.txt", 'r') as f:
                network_dict = json.load(f)
                network_dict[f"Networks-{network_num}"] = parameter_dict
            with open("Networks/network_parameters.txt", 'w') as f:
                json.dump(network_dict, f, indent=4)
        except (FileNotFoundError, JSONDecodeError):
            network_dict = {f"Networks-{network_num}": parameter_dict}
            with open("Networks/network_parameters_temp.txt", 'w') as f:
                json.dump(network_dict, f, indent=4)
            print("Error in saved dictionary. Wrote to new called '_temp'!")

    elif modes[mode] == "Compare":
        compare(network_num, plot)

    elif modes[mode] == "Test":
        result, avg_reward, confidence = check_solution(network_num)

    elif modes[mode] == "multi":
        print(f"Mode: multi")
        new_params = ({"discount": 0.2, "memory": 30000},
                      {"discount": 1, "memory": 30000},
                      {"discount": 0.99, "memory": 10000},
                      {"discount": 0.99, "memory": 50000})

        for params in new_params:
            print(f"Parameters: {params}")
            network_num = 1
            discount_factor = params["discount"]
            max_size = params["memory"]
            parameter_dict["discount_factor"] = discount_factor
            parameter_dict["max_size"] = max_size
            # Set name of model file to be saved
            actor_filename = f"actor-network-{network_num}"
            actor_filepath = f"Networks/{actor_filename}.pth"
            critic_filename = f"critic-network-{network_num}"
            critic_filepath = f"Networks/{critic_filename}.pth"
            while os.path.isfile(actor_filepath):
                network_num += 1
                actor_filename = f"actor-network-{network_num}"
                actor_filepath = f"Networks/{actor_filename}.pth"
                critic_filename = f"critic-network-{network_num}"
                critic_filepath = f"Networks/{critic_filename}.pth"
            filenames = (actor_filename, critic_filename)
            stopped_at = training_process(
                filenames, combined, early_stop=early_stop)
            parameter_dict["early_stop"] = early_stop
            parameter_dict["stopped_at"] = stopped_at
            result, avg_reward, confidence = check_solution(network_num)
            parameter_dict["Combined"] = combined
            parameter_dict["Results"] = {"Result": result,
                                         "avg_reward": avg_reward, "confidence": confidence}

            try:
                with open("Networks/network_parameters.txt", 'r') as f:
                    network_dict = json.load(f)
                    network_dict[f"Networks-{network_num}"] = parameter_dict
                with open("Networks/network_parameters.txt", 'w') as f:
                    json.dump(network_dict, f, indent=4)
            except (FileNotFoundError, JSONDecodeError):
                network_dict = {f"Networks-{network_num}": parameter_dict}
                with open(f"Networks/network_parameters_temp_{network_num}.txt", 'w') as f:
                    json.dump(network_dict, f, indent=4)
                print("Error in saved dictionary. Wrote to new called '_temp'!")
