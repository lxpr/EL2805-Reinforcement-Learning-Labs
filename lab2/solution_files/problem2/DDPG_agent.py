# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 2
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 26th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch.nn as nn
import torch


class Agent(object):
    ''' Base agent class

        Args:
            m (int): actions dimensionality

        Attributes:
            m (int): where we store the dimensionality of an action
    '''

    def __init__(self, m: int):
        self.m = m

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''

    def __init__(self, m: int):
        super(RandomAgent, self).__init__(m)

    def forward(self, state: np.ndarray) -> np.ndarray:
        ''' Compute a random action in [-1, 1]

            Returns:
                action (np.ndarray): array of float values containing the
                    action. The dimensionality is equal to self.m from
                    the parent class Agent.
        '''
        return np.clip(-1 + 2 * np.random.rand(self.m), -1, 1)


class ReplayBuffer:
    rng = np.random.default_rng()

    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size

    def sample(self, batch_size, combined=False):
        if combined:
            indices = self.rng.choice(self.size, batch_size - 1, replace=False)
            indices = np.append(indices, self.index - 1)
        else:
            indices = self.rng.choice(self.size, batch_size, replace=False)
        return [self.buffer[index] for index in indices]


class ActorNeuralNet(nn.Module):
    def __init__(self, neurons_1=400, neurons_2=200, input=3, output=1):
        super().__init__()  # This line needs to called to properly setup the network
        # Layer with 'input' inputs and `neurons` output
        self.linear1 = nn.Linear(input, neurons_1)
        self.act1 = nn.ReLU()  # Activation function
        self.linear2 = nn.Linear(neurons_1, neurons_2)
        self.act2 = nn.ReLU()  # Activation function
        # Layer with `neurons` inputs and 'output' outputs
        self.linear3 = nn.Linear(neurons_2, output)
        self.act3 = nn.Tanh()  # Activation function

    def forward(self, x):
        y1 = self.act1(self.linear1(x))

        y2 = self.act2(self.linear2(y1))

        y3 = self.linear3(y2)
        out = self.act3(y3)

        return out


class CriticNeuralNet(nn.Module):
    def __init__(self, neurons_1=400, neurons_2=200, states=3, actions=1):
        super().__init__()  # This line needs to called to properly setup the network
        # Layer with 'input' inputs and `neurons` output
        self.linear1 = nn.Linear(states, neurons_1)
        self.act1 = nn.ReLU()  # Activation function
        self.linear2 = nn.Linear(neurons_1 + actions, neurons_2)
        self.act2 = nn.ReLU()  # Activation function
        # Layer with `neurons` inputs and 'output' outputs
        self.linear3 = nn.Linear(neurons_2, 1)
        # self.act3 = nn.Tanh()  # Activation function

    def forward(self, x):
        s, a = x
        y1 = self.act1(self.linear1(s))

        x2 = torch.cat([y1, a], 1)

        y2 = self.act2(self.linear2(x2))

        out = self.linear3(y2)

        # out = self.act3(self.linear3(y2))

        return out


class DDPGAgent(Agent):
    """
    DDPG agent using 4 neural networks to make decisions
    """

    def __init__(self, neurons_1, neurons_2, m, dim_state, lr_actor, lr_critic, discount_factor, batch_size, clip_val,
                 max_size=10000, mu=0.15, sigma=0.2):
        super(DDPGAgent, self).__init__(m)
        self.rng = np.random.default_rng()
        self.actor = ActorNeuralNet(neurons_1=neurons_1, neurons_2=neurons_2, input=dim_state, output=m)
        self.actor_target = ActorNeuralNet(neurons_1=neurons_1, neurons_2=neurons_2, input=dim_state, output=m)
        self.critic = CriticNeuralNet(neurons_1=neurons_1, neurons_2=neurons_2, states=dim_state, actions=m)
        self.critic_target = CriticNeuralNet(neurons_1=neurons_1, neurons_2=neurons_2, states=dim_state, actions=m)
        self.update_actor_target()
        self.update_critic_target()
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.discount_factor = discount_factor
        self.buffer = ReplayBuffer(max_size=max_size)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        self.batch_size = batch_size
        self.clip_val = clip_val
        self.mu = mu
        self.sigma = sigma
        self.n_t = np.zeros((1, self.m))

    def forward(self, state):
        w = np.random.normal(0, self.sigma, (1, self.m))
        self.n_t = - self.mu * self.n_t + w
        # Create state tensor and feed to main network to generate action
        state_tensor = torch.tensor(np.array([state]), requires_grad=False)

        output = self.actor(state_tensor).detach().numpy()
        self.last_action = output[0] + self.n_t[0]
        self.last_action = np.clip(self.last_action, -1, 1)
        return self.last_action

    def learn(self, combined=False, actor_update=False):
        if self.buffer.size < 5 * self.batch_size:
            return 0
        # Perform learning step for the network
        # Reset gradients
        batch = self.buffer.sample(self.batch_size, combined=combined)

        # Get targets for the batch
        next_s = torch.tensor(np.array([exp['next_state']
                                        for exp in batch]), requires_grad=False)
        next_a = self.actor_target(next_s)
        next_a = next_a.detach().float()
        y = self.critic_target([next_s, next_a])

        # Indicator of whether episode finished with each experience
        d = torch.tensor(
            np.array([not exp['done'] for exp in batch]), requires_grad=False).reshape(-1, 1)
        # Reward for each experience
        r = torch.tensor(np.array([exp['reward']
                                   for exp in batch]), requires_grad=False).reshape(-1, 1)
        # Target values are r if the episode terminates in an experience,
        # and are r + gamma*max_a Q(s_t,a) if not
        y = r + self.discount_factor * d * y
        y = y.detach().float()
        self.opt_critic.zero_grad()
        # Get a batch of Q(s_t, a_t) for every (s_t, a_t) in batch
        s = torch.tensor(np.array([exp['state']
                                   for exp in batch]), requires_grad=True, dtype=torch.float32)
        a = torch.tensor(np.array([exp['action']
                                   for exp in batch]), requires_grad=True, dtype=torch.float32)
        x = self.critic([s, a])


        # Calculate MSE loss and perform backward step
        loss = nn.functional.mse_loss(x, y)

        loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_val)
        self.opt_critic.step()
        policy_loss = 0
        if actor_update:
            # Update theta
            s = torch.tensor(np.array([exp['state']
                                       for exp in batch]), requires_grad=True)
            self.opt_actor.zero_grad()
            action = self.actor(s)

            policy_loss = -torch.mean(self.critic([s, action]))
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_val)
            self.opt_actor.step()

        return loss, policy_loss

    def update_actor_target(self):
        # Copy parameters to target network
        self.actor_target.load_state_dict(self.actor.state_dict())

    def update_critic_target(self):
        # Copy parameters to target network
        self.critic_target.load_state_dict(self.critic.state_dict())

    def initialize_buffer(self, percentage, env):
        nr_steps = int(percentage * self.buffer.max_size)
        state = env.reset()
        m = len(env.action_space.high)
        agent = RandomAgent(m)
        for i in range(nr_steps):
            # Take a random action
            if i % 1000 == 0:
                print(i)
            action = agent.forward(state)
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)
            # Append experience to buffer
            experience = {'state': state, 'action': action,
                          'reward': reward, 'next_state': next_state, 'done': done}
            self.buffer.append(experience)
            state = next_state
            if done:
                env.reset()
        print('Initialization finished')
