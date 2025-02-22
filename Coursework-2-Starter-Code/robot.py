####################################
#      YOU MAY EDIT THIS FILE      #
# ALL OF YOUR CODE SHOULD GO HERE #
####################################

# Imports from external libraries
import numpy as np
import torch, random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
from collections import deque

# Imports from this project
# You should not import any other modules, including config.py
# If you want to create some configuration parameters for your algorithm, keep them within this robot.py file
from config import SAC_CONFIG
import constants
from graphics import VisualisationLine

# Configure matplotlib for interactive mode
plt.ion()

# CONFIGURATION PARAMETERS. Add whatever configuration parameters you like here.
# Remember, you will only be submitting this robot.py file, no other files.

class Actor(nn.Module):
    def __init__(self, hidden_dim=SAC_CONFIG['hidden_dim']):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(constants.OBSERVATION_DIMENSION, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        # Output: mean and log_std for action distribution
        self.mean_layer = nn.Linear(hidden_dim, constants.ACTION_DIMENSION)
        self.log_std_layer = nn.Linear(hidden_dim, constants.ACTION_DIMENSION)
        
    def forward(self, obs):
        y = self.model(obs)
        mean = self.mean_layer(y)
        log_std = self.log_std_layer(y)
        # Clip log_std for stability
        log_std = torch.clamp(log_std, -2, 2)
        std = log_std.exp()
        z = torch.randn_like(mean)
        action = mean + std * z
        log_prob = (-0.5 * z.pow(2) - log_std).sum(dim=-1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, hidden_dim=SAC_CONFIG['hidden_dim']):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(constants.OBSERVATION_DIMENSION+constants.ACTION_DIMENSION, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Q-value output
        )
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.model(x)

class SoftActorCritic:
    def __init__(self):
        self.replay_buffer = deque([], SAC_CONFIG['buffer_size'])
        self.actor = Actor()
        self.critics = (Critic(), Critic())
        self.critic_targets = (Critic(), Critic())
        for i in [0,1]:
            self.critic_targets[i].load_state_dict(self.critics[i].state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=SAC_CONFIG['lr_actor'])
        self.critic_optimizers = [torch.optim.Adam(self.critics[i].parameters(), lr=SAC_CONFIG['lr_critic']) for i in [0,1]]
        self.target_entropy = -constants.ACTION_DIMENSION
        self.log_alpha = torch.zeros(1, 1, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=SAC_CONFIG['lr_critic'])
    
    def replay_sample(self, batch_size=SAC_CONFIG['batch_size']):
        batch = random.sample(self.replay_buffer, batch_size)
        return tuple(torch.stack(t) for t in zip(*batch))
    
    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.Tensor(obs).unsqueeze(0)
            action, _ = self.actor.forward(obs)
            return action.squeeze().numpy()

    def update(self, batch_size=SAC_CONFIG['batch_size']):
        # Sample from replay buffer
        obs, action, next_obs, reward, done = self.replay_sample(batch_size)
        with torch.no_grad():
            next_action, next_log_prob = self.actor.forward(next_obs)
            target_Q = [self.critics[i](next_obs, next_action) for i in [0,1]]
            target_Q = torch.min(*target_Q)
            alpha = self.log_alpha.exp()
            target_Q = reward + (1-done) * SAC_CONFIG['discount'] * (target_Q - alpha*next_log_prob)
        
        for i in [0,1]:
            self.critic_optimizers[i].zero_grad()
            critic_loss = F.mse_loss(self.critics[i](obs, action), target_Q)
            critic_loss.backward()
            self.critic_optimizers[i].step()

        new_action, log_prob = self.actor.forward(obs)
        Q = [self.critics[i](obs, new_action) for i in [0,1]]
        Q = torch.min(*Q)
        actor_loss = (alpha * log_prob - Q).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for i in [0,1]:
            for param, target_param in zip(self.critics[i].parameters(), self.critic_targets[i].parameters()):
                tau = SAC_CONFIG['tau']
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:
    # Initialise a new robot
    def __init__(self):
        # The environment (only available during development mode)
        self.environment = None
        # A list of visualisations which will be displayed on the bottom half of the window
        self.visualisation_lines = []
        self.brain = SoftActorCritic()
        self.episode_steps = 0
        self.prev_dist = None

    # Get the next training action
    def training_action(self, obs, money):
        action_value = np.zeros(constants.ACTION_DIMENSION)            
        if self.episode_steps < SAC_CONFIG['init_steps']:
            action_value = np.random.uniform(-constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE, 2)
        else:
            if self.episode_steps > 1500 or self.brain.replay_buffer[-1][4].numpy() == 1:
                # Episode done, reset or end
                action_type = 2 if money > constants.INIT_MONEY*0.7 else 4
                print("Action type:", action_type)
                self.episode_steps = 0
                reset_state = np.array([0.05,np.random.uniform(0, constants.ENVIRONMENT_HEIGHT)])
                return action_type, reset_state
            elif len(self.brain.replay_buffer) > SAC_CONFIG['batch_size']:
                # Update SAC model every n steps
                if self.episode_steps % 100 == 0:
                    print(self.episode_steps, money)
                for _ in range(50):
                    self.brain.update()
            action_value = self.brain.select_action(obs)
        return 1, action_value
        # return 1, np.array([constants.MAX_ACTION_MAGNITUDE, 0])

    # Get the next testing action
    def testing_action(self, obs):
        # Random action
        if self.episode_steps < SAC_CONFIG['init_steps']/4:
            return np.random.uniform(-constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE, 2)
        elif len(self.brain.replay_buffer) > SAC_CONFIG['batch_size']:
            self.brain.update()
        action = self.brain.select_action(obs)
        return action

    # Receive a transition
    def receive_transition(self, obs, action, next_obs, reward):
        # Reward shaping
        shaped_reward = 100/(1 - reward) - 40
        # shaped_reward = -100*reward - 10
        # shaped_reward = reward+1
        done = 0
        if reward >= -0.01 or (hasattr(self, "environment") and self.environment.state[0] >= 1.95):
            done = 1
        elif self.prev_dist is not None:
            shaped_reward += 100*abs(reward-self.prev_dist) - 1
        self.prev_dist = reward
        self.episode_steps += 1
        self.brain.replay_buffer.append((torch.Tensor(obs), 
                                         torch.Tensor(action),
                                         torch.Tensor(next_obs), 
                                         torch.Tensor([shaped_reward]),
                                         torch.Tensor([done])))
    # Receive a new demonstration
    def receive_demo(self, demo):
        pass


if __name__ == "__main__":
    # print(torch.Tensor([np.zeros(2), np.ones(2)]))
    # xs = np.arange(0, -2, -0.01)
    # ys = 100/(1-xs)-40
    # plt.plot(xs, ys)
    # plt.savefig("function.png")
    print(np.exp(2))
