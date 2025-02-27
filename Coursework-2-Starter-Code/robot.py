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
from config import RL_CONFIG
import constants
from graphics import VisualisationLine

# Configure matplotlib for interactive mode
plt.ion()

# CONFIGURATION PARAMETERS. Add whatever configuration parameters you like here.
# Remember, you will only be submitting this robot.py file, no other files.

class RLAlgorithm:
    def __init__(self):
        self.replay_buffer = deque([], RL_CONFIG['buffer_size'])
        self.demo_buffer = deque([], RL_CONFIG['buffer_size'])
    
    def replay_sample(self): 
        if len(self.replay_buffer) < RL_CONFIG['buffer_size']:
            batch = self.replay_buffer
        else:
            recent = int(RL_CONFIG['buffer_size']/4)
            batch = self.replay_buffer[-recent:]
            batch = random.sample(self.replay_buffer[:-recent], RL_CONFIG['buffer_size'] - recent) + batch
        return tuple(torch.stack(t) for t in zip(*batch))


class DQN_Learning(RLAlgorithm):
    def __init__(self):
        super().__init__()
        self.q_network = self._build_q_model()
        self.d_network = self._build_dynamic_model()
        self.target_network = self._build_q_model()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=RL_CONFIG['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()
        self.epsilon = 0
    @staticmethod
    def _build_q_model():
        """Builds a neural network model for Q-function approximation"""
        hidden = RL_CONFIG['hidden_dim']
        model = nn.Sequential(
            nn.Linear(constants.OBSERVATION_DIMENSION, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, RL_CONFIG['num_actions']),
            nn.Sigmoid()
        )
        return model
    @staticmethod
    def _build_dynamic_model():
        hidden = RL_CONFIG['hidden_dim']
        model = nn.Sequential(
            nn.Linear(constants.OBSERVATION_DIMENSION+1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, constants.OBSERVATION_DIMENSION+1),
        )
        return model
    def select_action(self, obs, raw=False):
        """Select action using epsilon-greedy policy"""
        obs = torch.FloatTensor(obs).unsqueeze(0)
        self.q_network.eval()
        with torch.no_grad():
            q_values = self.q_network(obs)
        if raw:
            return q_values
        # print(q_values)
        det = np.random.rand()
        if det > self.epsilon:
            return q_values.argmax().item()
        index = int(q_values.shape[1] * (self.epsilon-det)/self.epsilon)
        _, indices = torch.sort(q_values[0])
        return indices[index].item()
    def decay_epsilon(self):
        """Decay exploration rate"""
        if self.epsilon > RL_CONFIG['epsilon_min']:
            self.epsilon *= RL_CONFIG['epsilon_decay']

    def update(self, epochs=30):
        if not self.epsilon:
            self.epsilon = RL_CONFIG['epsilon']
        self.q_network.train()
        for _ in range(epochs):
            obs, actions, next_obs, rewards, dones = self.replay_sample()
            current_q_values = self.q_network(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
            # Compute target Q values
            with torch.no_grad():
                next_q_values = self.target_network(next_obs).max(1)[0]
                target_q_values = rewards + (1 - dones) * RL_CONFIG['gamma'] * next_q_values
            
            # Compute loss and update weights
            self.optimizer.zero_grad()
            loss = self.criterion(current_q_values, target_q_values)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.optimizer.step()
        self.decay_epsilon()
        return loss.item()
    
    def update_target_network(self, episodes):
        """Update target network at specified frequency"""
        if episodes % RL_CONFIG['target_update_freq'] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def behavior_cloning(self, epochs=800):
        observations, actions = tuple(torch.stack(t) for t in zip(*self.demo_buffer))
        dataset = torch.utils.data.TensorDataset(observations, actions)
        loader = torch.utils.data.DataLoader(dataset, batch_size=RL_CONFIG['batch_size'], shuffle=True)
        self.q_network.train()
        for epoch in range(epochs):
            for batch_obs, batch_actions in loader:
                q_values = self.q_network(batch_obs)
                demo = torch.zeros_like(q_values)
                demo.scatter_(1, batch_actions.reshape(-1,1), 1.0)
                self.optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(q_values, demo)
                loss.backward()
                self.optimizer.step()
    
    def offline_training(self, epochs):
        print("Extra training")
        for epoch in range(epochs):
            self.update(200)
            self.update_target_network(0)
        self.epsilon = 0

# The Robot class (which could be called "Agent") is the "brain" of the robot, and is used to decide what action to execute in the environment
class Robot:
    # Initialise a new robot
    def __init__(self):
        # The environment (only available during development mode)
        self.environment = None
        self.prev_state = None
        self.prev_reward = None
        # A list of visualisations which will be displayed on the bottom half of the window
        self.visualisation_lines = []
        self.brain = DQN_Learning()
        self.done = 0
        self.episode_steps = 0
        self.demo_required = 3
    
    @staticmethod
    def action_to_vector(action):
        h_actions = RL_CONFIG['num_actions']/2
        rad = np.pi * (action - h_actions+1)/h_actions
        dx = constants.MAX_ACTION_MAGNITUDE * np.cos(rad)
        dy = constants.MAX_ACTION_MAGNITUDE * np.sin(rad)
        return np.array([dx, dy])
    @staticmethod
    def vector_to_action(vector):
        h_actions = RL_CONFIG['num_actions']/2
        rad = np.arctan2(vector[1], vector[0])
        action = np.ceil(h_actions * rad/np.pi + h_actions-1)
        return torch.tensor(max(action, 0), dtype=torch.int64)

    # Get the next training action
    def training_action(self, obs, money):
        if self.episode_steps == 0 and money < 15:
            # Extra training using model prediction
            self.brain.offline_training(50)
            return 4, np.zeros(2)
            
        elif self.episode_steps > 800 or self.done:
            # Episode done, reset or end
            self.episode_steps = 0
            self.brain.epsilon = 0
            self.prev_state = None
            self.prev_reward = None
            reset_state = np.array([0.05,np.random.uniform(0, constants.ENVIRONMENT_HEIGHT)])
            self.done = 0
            return 2, reset_state
        
        action_value = []
        if self.demo_required:
            print("Collect expert demonstration")
            return 3, np.array([0, RL_CONFIG['demo_steps']])
        else:
            if len(self.brain.replay_buffer) > RL_CONFIG['demo_steps']:
                self.brain.update()
                self.brain.update_target_network(len(self.brain.replay_buffer))
                if self.episode_steps % 50 == 0:
                    print(self.episode_steps, money)
            action_value = Robot.action_to_vector(self.brain.select_action(obs))
        return 1, action_value

    # Get the next testing action
    def testing_action(self, obs):
        # Random action
        self.brain.replay_buffer.clear()
        self.brain.demo_buffer.clear()
        if len(self.brain.replay_buffer) > RL_CONFIG['batch_size']:
            self.brain.update()
        action = self.brain.select_action(obs)
        return Robot.action_to_vector(action)

    # Receive a transition
    def receive_transition(self, obs, action, next_obs, reward):
        shaped_reward = reward-5
        if self.environment is not None:
            if self.prev_state is not None:
                x1, y1 = tuple(self.prev_state)
                x2, y2 = tuple(self.environment.state)
                real_action = np.clip(action, -constants.MAX_ACTION_MAGNITUDE, constants.MAX_ACTION_MAGNITUDE)
                xd, yd = (self.environment.state + real_action * 5)
                xr, yr = (self.environment.state - real_action * 5)
                if len(self.visualisation_lines) >= 1:
                    self.visualisation_lines.pop()
                    self.visualisation_lines.pop()
                self.visualisation_lines.extend([VisualisationLine(x1, y1, x2, y2, (100,0,200), 0.005),
                    VisualisationLine(x2, y2, xd, yd, (20,200,50), 0.01),
                    VisualisationLine(x2, y2, xr, yr, (200,20,50), 0.01)])
            self.prev_state = self.environment.state
            if self.environment.state[0] >= 1.95:
                self.done = 1
        elif reward >= -0.01:
            self.done = 1
            shaped_reward += 100
        self.episode_steps += 1
        if self.prev_reward is not None and abs( reward - self.prev_reward ) < 1e-4:
            shaped_reward -= 12
        else:
            for checkpoint in np.arange(-1.8, 0, -0.2):
                if reward > checkpoint:
                    shaped_reward += 1
        self.prev_reward = reward
        self.brain.replay_buffer.append((torch.Tensor(obs), 
                                         Robot.vector_to_action(action),
                                         torch.Tensor(next_obs), 
                                         torch.tensor(shaped_reward),
                                         torch.tensor(self.done)))
        # if self.episode_steps % 400 == 0:
        #     self.demo_required = True
    # Receive a new demonstration
    def receive_demo(self, demo):
        self.brain.demo_buffer.extend([
            (torch.Tensor(obs), Robot.vector_to_action(action)) for obs, action in demo
        ])
        print("Demo received", len(demo), len(self.brain.demo_buffer))
        self.brain.behavior_cloning()
        self.demo_required -= 1     


if __name__ == "__main__":
    # print(torch.Tensor([np.zeros(2), np.ones(2)]))
    # xs = np.arange(0, -2, -0.01)
    # ys = 100/(1-xs)-40
    # plt.plot(xs, ys)
    # plt.savefig("function.png")
    print(np.floor(-0.5))
