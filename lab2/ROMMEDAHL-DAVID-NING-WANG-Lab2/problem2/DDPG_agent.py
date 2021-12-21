# Load packages
import numpy as np
import torch.nn as nn
import torch

# Make new critic network for DDPG problem


class CriticNet(nn.Module):
    def __init__(self, input=8):
        super().__init__()  # This line needs to called to properly setup the network

        self.linear_1 = nn.Linear(input, 400)
        self.act_1 = nn.ReLU()

        self.linear_2 = nn.Linear(402, 200)
        self.act_2 = nn.ReLU()

        self.linear_3 = nn.Linear(200, 1)

    # Forward method to produce action and value

    def forward(self, s, action):

        x_1 = self.act_1(self.linear_1(s))

        # Concatenate actions and output from first critic layer
        critic_input = torch.cat((x_1, action), dim=1)

        # Value of actions and state
        x_2 = self.act_2(self.linear_2(critic_input))

        # Output value
        value = self.linear_3(x_2)

        return value

# Make new actor network for DDPG problem


class ActorNet(nn.Module):
    def __init__(self, input=8, output=2):
        super().__init__()  # This line needs to called to properly setup the network

        # Dimensionality of output
        self.m = output

        self.linear_1 = nn.Linear(input, 400)
        self.act_1 = nn.ReLU()

        self.linear_2 = nn.Linear(400, 200)
        self.act_2 = nn.ReLU()

        self.linear_3 = nn.Linear(200, self.m)
        self.act_3 = nn.Tanh()

    # Forward method to produce action and value
    def forward(self, s):

        # Produce ations from state
        x_1 = self.act_1(self.linear_1(s))

        x_2 = self.act_2(self.linear_2(x_1))

        action = self.act_3(self.linear_3(x_2))

        return action


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
        batch = [self.buffer[index] for index in indices]
        return zip(*batch)


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

    def forward(self, s, a):
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
        self.actor = ActorNeuralNet(
            neurons_1=neurons_1, neurons_2=neurons_2, input=dim_state, output=m)
        self.actor_target = ActorNeuralNet(
            neurons_1=neurons_1, neurons_2=neurons_2, input=dim_state, output=m)
        self.critic = CriticNeuralNet(
            neurons_1=neurons_1, neurons_2=neurons_2, states=dim_state, actions=m)
        self.critic_target = CriticNeuralNet(
            neurons_1=neurons_1, neurons_2=neurons_2, states=dim_state, actions=m)
        self.update_actor_target()
        self.update_critic_target()
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.discount_factor = discount_factor
        self.buffer = ReplayBuffer(max_size=max_size)
        self.opt_actor = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr_actor)
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr_critic)
        self.batch_size = batch_size
        self.clip_val = clip_val
        self.mu = mu
        self.sigma = sigma
        self.n_t = np.zeros((1, self.m))

    def forward(self, state):
        # w = np.random.normal(0, self.sigma, (1, self.m))
        # self.n_t = - self.mu * self.n_t + w
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
        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size)

        # # Get targets for the batch
        next_s = torch.tensor(
            next_states, requires_grad=False, dtype=torch.float32)
        next_a = self.actor_target(next_s)
        next_a = next_a.float()
        y = self.critic_target(next_s, next_a)

        # Indicator of whether episode finished with each experience
        d = torch.tensor(
            dones, requires_grad=False, dtype=torch.float32).reshape(-1, 1)
        # Reward for each experience
        r = torch.tensor(rewards, requires_grad=False,
                         dtype=torch.float32).reshape(-1, 1)
        # Target values are r if the episode terminates in an experience,
        # and are r + gamma*max_a Q(s_t,a) if not
        y = r + self.discount_factor * (1 - d) * y
        # y = y.detach().float()
        self.opt_critic.zero_grad()
        # Get a batch of Q(s_t, a_t) for every (s_t, a_t) in batch
        s = torch.tensor(states, requires_grad=True, dtype=torch.float32)
        a = torch.tensor(actions, requires_grad=True, dtype=torch.float32)
        x = self.critic(s, a)

        # Calculate MSE loss and perform backward step
        loss = nn.functional.mse_loss(x, y)

        loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(), max_norm=self.clip_val)
        self.opt_critic.step()
        policy_loss = 0
        if actor_update:
            # Update theta
            s = torch.tensor(states, requires_grad=True, dtype=torch.float32)
            self.opt_actor.zero_grad()
            action = self.actor(s)

            policy_loss = -torch.mean(self.critic(s, action))
            policy_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor.parameters(), max_norm=self.clip_val)
            self.opt_actor.step()
        return loss, policy_loss

    def noise(self):
        self.n_t = -self.mu * self.n_t + \
            np.random.multivariate_normal(np.zeros(self.m), pow(
                self.sigma, 2) * np.identity(self.m))

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
            experience = (state, action, reward, next_state, done)
            self.buffer.append(experience)
            state = next_state
            if done:
                env.reset()
        print('Initialization finished')
