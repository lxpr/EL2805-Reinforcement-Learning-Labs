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
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn


class Agent(object):
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.last_action = None

    def forward(self, state: np.ndarray):
        ''' Performs a forward computation '''
        pass

    def backward(self):
        ''' Performs a backward pass on the network '''
        pass


class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


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


class NeuralNet(nn.Module):
    def __init__(self, neurons=32, input=3, output=1):
        super().__init__()  # This line needs to called to properly setup the network
        # Layer with 'input' inputs and `neurons` output
        self.linear1 = nn.Linear(input, neurons)
        self.act1 = nn.ReLU()  # Activation function
        self.linear2 = nn.Linear(neurons, neurons)
        self.act2 = nn.ReLU()  # Activation function
        # Layer with `neurons` inputs and 'output' outputs
        self.linear3 = nn.Linear(neurons, output)

    def forward(self, x):

        y1 = self.act1(self.linear1(x))

        y2 = self.act2(self.linear2(y1))

        out = self.linear3(y2)

        return out

        out = self.linear3(y2)

        return out


class DQNAgent(Agent):
    """
    DQN agent using two neural networks to make decisions
    """

    def __init__(self, neurons, n_actions, dim_state, lr, discount_factor, batch_size, clip_val, max_size=10000):
        super(DQNAgent, self).__init__(n_actions)
        self.rng = np.random.default_rng()
        self.network = NeuralNet(neurons, input=dim_state, output=n_actions)
        self.target = NeuralNet(neurons, input=dim_state, output=n_actions)
        self.update_target()
        self.lr = lr
        self.discount_factor = discount_factor
        self.buffer = ReplayBuffer(max_size=max_size)
        self.opt = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.batch_size = batch_size
        self.clip_val = clip_val

    def forward(self, state, epsilon):
        self.epsilon = epsilon
        if self.rng.choice([0, 1], p=[1-self.epsilon, self.epsilon]):
            self.last_action = self.rng.choice(self.n_actions)
            return self.last_action
        # Create state tensor and feed to main network to generate action
        state_tensor = torch.tensor(np.array([state]), requires_grad=False)

        output_tensor = self.network(state_tensor)

        self.last_action = output_tensor.max(1)[1].item()
        return self.last_action

    def learn(self, combined=False):
        if self.buffer.size < 5*self.batch_size:
            return 0
        # Perform learning step for the network
        # Reset gradients
        batch = self.buffer.sample(self.batch_size, combined=combined)

        # Get targets for the batch
        y = torch.tensor(np.array([exp['next_state']
                         for exp in batch]), requires_grad=False)
        y = self.target(y)

        # Indicator of whether episode finished with each experience
        d = torch.tensor(
            np.array([not exp['done'] for exp in batch]), requires_grad=False)
        # Reward for each experience
        r = torch.tensor(np.array([exp['reward']
                         for exp in batch]), requires_grad=False)
        # Target values are r if the episode terminates in an experience,
        # and are r + gamma*max_a Q(s_t,a) if not
        y = r + self.discount_factor*d*y.max(1)[0]
        y = y.float().detach()

        self.opt.zero_grad()

        # Get a batch of Q(s_t, a_t) for every (s_t, a_t) in batch
        x = torch.tensor(np.array([exp['state']
                         for exp in batch]), requires_grad=True)
        x = self.network(x)
        x = x.gather(1, torch.tensor([[exp['action']]
                     for exp in batch])).reshape(-1)
        #x = torch.tensor([x[i, exp['action']] for i, exp in enumerate(batch)], requires_grad=True)

        # Calculate MSE loss and perform backward step
        loss = nn.functional.mse_loss(x, y)

        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.clip_val)
        self.opt.step()
        return loss

    def update_target(self):
        # Copy parameters to target network
        self.target.load_state_dict(self.network.state_dict())

    def initialize_buffer(self, percentage, env):
        nr_steps = int(percentage*self.buffer.max_size)
        state = env.reset()
        for i in range(nr_steps):
            # Take a random action
            action = self.forward(state, epsilon=1)
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise
            next_state, reward, done, _ = env.step(action)
            # Append experience to buffer
            experience = {'state': state, 'action': action,
                          'reward': reward, 'next_state': next_state, 'done': done}
            self.buffer.append(experience)
            state = next_state