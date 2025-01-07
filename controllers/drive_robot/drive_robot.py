import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from robot_desc import Rob
from datetime import datetime
import os

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class QNetwork(nn.Module):
    def __init__(self, state_size, actions_per_actuator, num_actuators):
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.actions_per_actuator = actions_per_actuator
        self.num_actuators = num_actuators
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Separate output layers for each actuator
        self.output_layers = nn.ModuleList([
            nn.Linear(128, actions_per_actuator) for _ in range(num_actuators)
        ])
        
        # Initialize weights
        for layer in [self.fc1, self.fc2] + list(self.output_layers):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # Move model to GPU if available
        self.to(device)
        
    def forward(self, x):
        if torch.isnan(x).any():
            print("NaN detected in input!")
            x = torch.nan_to_num(x, 0.0)
        
        x = torch.clamp(x, -10, 10)
        
        # Shared features
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        # Separate outputs for each actuator
        outputs = [layer(x) for layer in self.output_layers]
        return torch.stack(outputs, dim=1)

class RobotDQNAgent:
    def __init__(self, rob, state_size, actions_per_actuator, active_sensor_idx=0, historical_window=5, target_sensor_weight=0.5):
        self.rob = rob
        self.state_size = state_size
        self.num_actuators = len(rob.get_actuators())
        self.actions_per_actuator = actions_per_actuator
        self.active_sensor_idx = active_sensor_idx
        
        # Validate sensor index
        num_sensors = len(rob.get_sensors())
        if not 0 <= active_sensor_idx < num_sensors:
            raise ValueError(f"Invalid active_sensor_idx {active_sensor_idx}. Must be between 0 and {num_sensors-1}")
        
        # DQN hyperparameters
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Increased gamma to value future rewards more
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999
        self.learning_rate = 0.001
        self.batch_size = 32
        self.max_grad_norm = 1.0

        # Reward structure
        self.target_sensor_weight = target_sensor_weight

        self.steps_per_episode = 1000
        
        # Track histories for all sensors
        self.historical_window = historical_window
        self.reading_history = deque(maxlen=10)  # For active sensor potential calculation
        self.sensor_histories = {i: deque(maxlen=100) for i in range(num_sensors)}
        self.sensor_stats = {i: {'min': None, 'max': None, 'mean': None, 'std': None} 
                           for i in range(num_sensors)}
        
        # Initialize networks
        self.q_network = QNetwork(state_size, actions_per_actuator, self.num_actuators)
        self.target_network = QNetwork(state_size, actions_per_actuator, self.num_actuators)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.action_space = np.linspace(-1, 1, actions_per_actuator)

    def get_state(self):
        """Get current state (actuator positions) with validation"""
        state = np.array([act.get_position() for act in self.rob.get_actuators()])
        
        # Check for invalid values
        if np.isnan(state).any():
            print("Warning: NaN detected in state!")
            state = np.nan_to_num(state, 0.0)
        
        # Normalize state values
        state = np.clip(state, -10, 10)
        return state
    
    def get_reward_with_metrics(self, initial_readings, final_readings):
        """Calculate reward and return metrics data"""
        # Normalize readings
        target_initial = self.normalize_reading(self.active_sensor_idx, initial_readings[self.active_sensor_idx])
        target_final = self.normalize_reading(self.active_sensor_idx, final_readings[self.active_sensor_idx])
        
        # Calculate historical difference
        historical_window = self.historical_window
        if len(self.reading_history) >= historical_window:
            historical_diff = target_final - self.reading_history[-historical_window]
        else:
            historical_diff = target_final - target_initial
        
        # Calculate stability penalty and normalize other sensors
        stability_penalty = 0
        normalized_others = {}
        for i, (init_reading, final_reading) in enumerate(zip(initial_readings, final_readings)):
            if i != self.active_sensor_idx:
                init_norm = self.normalize_reading(i, init_reading)
                final_norm = self.normalize_reading(i, final_reading)
                normalized_others[i] = final_norm
                stability_penalty += abs(final_norm - init_norm)
        
        # Calculate final reward
        sensor_weight = self.target_sensor_weight
        stability_weight = 1 - target_sensor_weight
        reward = (sensor_weight * historical_diff - 
                stability_weight * stability_penalty)
        
        # Update history
        self.reading_history.append(target_final)
        
        # Return reward and metrics
        metrics_data = {
            'normalized_target': target_final,
            'historical_diff': historical_diff,
            'stability_penalty': stability_penalty,
            'normalized_others': normalized_others
        }
        
        return np.clip(reward, -10, 10), metrics_data

    def update_sensor_stats(self, sensor_idx, reading):
        """Update running statistics for sensor normalization"""
        history = self.sensor_histories[sensor_idx]
        history.append(reading)
        
        if len(history) >= 10:  # Wait for some history to accumulate
            stats = self.sensor_stats[sensor_idx]
            readings_array = np.array(history)
            stats['min'] = np.min(readings_array)
            stats['max'] = np.max(readings_array)
            stats['mean'] = np.mean(readings_array)
            stats['std'] = np.std(readings_array) if len(history) > 1 else 1.0
    
    def get_all_sensor_readings(self):
        """Get readings from all sensors"""
        try:
            readings = []
            for i, sensor in enumerate(self.rob.get_sensors()):
                reading = sensor.get_reading()
                if np.isnan(reading):
                    print(f"Warning: NaN reading from sensor {i}")
                    reading = 0.0
                readings.append(reading)
                self.update_sensor_stats(i, reading)
            return readings
        except Exception as e:
            print(f"Error reading sensors: {e}")
            return [0.0] * len(self.rob.get_sensors())
   
    def normalize_reading(self, sensor_idx, reading):
        """Normalize sensor reading based on historical statistics"""
        stats = self.sensor_stats[sensor_idx]
        
        # If we don't have enough history, use the current reading
        # to create initial statistics
        if stats['mean'] is None:
            stats['mean'] = reading
            stats['std'] = 1.0  # Use a default standard deviation
            stats['min'] = reading
            stats['max'] = reading
        
        # Z-score normalization with clipping
        if stats['std'] > 0:
            normalized = (reading - stats['mean']) / stats['std']
        else:
            normalized = reading - stats['mean']
            
        return np.clip(normalized, -3, 3)  # Clip to 3 standard deviations
    
    def act(self, state):
        """Choose an action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Random actions for each actuator
            return [random.randrange(self.actions_per_actuator) 
                   for _ in range(self.num_actuators)]
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action_values = self.q_network(state_tensor)
                return torch.argmax(action_values.squeeze(), dim=1).cpu().tolist()
    
    def remember(self, state, actions, reward, next_state):
        """Store experience with multiple actions"""
        self.memory.append((state, actions, reward, next_state))
    
    def replay(self):
        """Experience replay with multiple actions"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*minibatch)
        
        # Convert to tensors and move to GPU
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        
        if torch.isnan(states).any() or torch.isnan(next_states).any():
            print("NaN detected in batch data!")
            return
        
        try:
            # Current Q values for each actuator
            current_q_all = self.q_network(states)
            current_q = torch.gather(current_q_all, 2, actions.unsqueeze(2)).squeeze(2)
            
            # Next Q values
            with torch.no_grad():
                next_q_all = self.target_network(next_states)
                next_q = next_q_all.max(2)[0]
            
            # Target Q values (same reward applied to all actuators)
            target_q = rewards.unsqueeze(1) + self.gamma * next_q
            
            # Compute loss across all actuators
            loss = nn.MSELoss()(current_q, target_q)
            
            if torch.isnan(loss):
                print("NaN loss detected!")
                return
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        except Exception as e:
            print(f"Error during replay: {e}")
    
    def update_target_network(self):
        """Update target network periodically"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train_episode(self, max_steps=1000):
        """Training episode with focus on active sensor"""
        total_reward = 0
        state = self.get_state()
        episode_readings = []
        self.action_history = []
        
        for step in range(max_steps):
            initial_readings = self.get_all_sensor_readings()
            actions = self.act(state)
            self.action_history.append(actions)
            
            # Apply actions to actuators
            for actuator, action_idx in zip(self.rob.get_actuators(), actions):
                motor_intensity = self.action_space[action_idx]
                actuator.apply_intensity(motor_intensity * actuator.max_speed)
            
            # Run simulation with smaller steps for stability
            for _ in range(100):
                self.rob.step()
            
            # Get new state and calculate reward
            next_state = self.get_state()
            final_readings = self.get_all_sensor_readings()
            reward = self.get_reward(initial_readings, final_readings)
            
            # Store experience and train
            self.remember(state, actions, reward, next_state)
            self.replay()
            
            total_reward += reward
            state = next_state
            episode_readings.append(final_readings[self.active_sensor_idx])
            
            if step % 100 == 0:
                self.update_target_network()
                active_reading = final_readings[self.active_sensor_idx]
                stability_penalty = sum(abs(self.normalize_reading(i, r)) 
                                     for i, r in enumerate(final_readings) 
                                     if i != self.active_sensor_idx)
                print(f"Step {step}, Active Sensor ({self.active_sensor_idx}) Reading: {active_reading:.3f}, "
                      f"Reward: {reward:.3f}, Stability Penalty: {stability_penalty:.3f}")
        
        return total_reward, episode_readings

class TrainingMetrics:
    def __init__(self):
        # Initialize lists to store metrics
        self.data = {
            'episode': [],
            'step': [],
            'raw_reading': [],
            'normalized_reading': [],
            'stability_penalty': [],
            'historical_diff': [],
            'reward': [],
            'epsilon': [],
            'total_episode_reward': []
        }
        # Dynamically add other sensor columns as needed
        self.other_sensor_columns = set()
        
    def update(self, episode, step, raw_reading, norm_reading, stability, hist_diff, 
               raw_others, norm_others, reward, epsilon):
        # Add main metrics
        self.data['episode'].append(episode)
        self.data['step'].append(step)
        self.data['raw_reading'].append(raw_reading)
        self.data['normalized_reading'].append(norm_reading)
        self.data['stability_penalty'].append(stability)
        self.data['historical_diff'].append(hist_diff)
        self.data['reward'].append(reward)
        self.data['epsilon'].append(epsilon)
        self.data['total_episode_reward'].append(0)  # Will update at episode end
        
        # Add other sensor data
        for idx, raw_val in raw_others.items():
            raw_col = f'sensor_{idx}_raw'
            norm_col = f'sensor_{idx}_norm'
            self.other_sensor_columns.add(raw_col)
            self.other_sensor_columns.add(norm_col)
            
            if raw_col not in self.data:
                self.data[raw_col] = []
            if norm_col not in self.data:
                self.data[norm_col] = []
                
            self.data[raw_col].append(raw_val)
            self.data[norm_col].append(norm_others[idx])
    
    def end_episode(self, episode, total_reward):
        # Update total_episode_reward for all steps in this episode
        episode_mask = [i == episode for i in self.data['episode']]
        for i, is_episode in enumerate(episode_mask):
            if is_episode:
                self.data['total_episode_reward'][i] = total_reward
    
    def save_metrics(self, output_dir, timestamp):
        """Save all metrics to a single CSV file"""
        import pandas as pd
        
        # Convert data to DataFrame
        df = pd.DataFrame(self.data)
        
        # Sort columns for better organization
        fixed_columns = ['episode', 'step', 'raw_reading', 'normalized_reading', 
                        'stability_penalty', 'historical_diff', 'reward', 
                        'epsilon', 'total_episode_reward']
        other_columns = sorted(list(self.other_sensor_columns))
        df = df[fixed_columns + other_columns]
        
        # Save to CSV
        csv_file = os.path.join(output_dir, f'training_metrics_{timestamp}.csv')
        df.to_csv(csv_file, index=False)
        print(f"Saved all metrics to {csv_file}")

def save_model(agent, output_dir, model_name="robot_dqn_model"):
    """
    Save the Q-network of the agent to the specified directory.

    :param agent: RobotDQNAgent instance
    :param output_dir: Directory where to save the model
    :param model_name: Name to give the saved model file
    """
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}.pth")
    # Move model to CPU before saving
    agent.q_network.cpu()
    torch.save(agent.q_network.state_dict(), model_path)
    # Move model back to original device
    agent.q_network.to(device)
    print(f"Model saved to {model_path}")

def train_robot(rob, active_sensor_idx=0, num_episodes=100, historical_window=5, target_sensor_weight=0.5, output_dir='output'):
    rob.setup()
    state_size = len(rob.get_actuators())
    actions_per_actuator = 2  # Number of discrete actions per actuator
    
    agent = RobotDQNAgent(rob, state_size, actions_per_actuator, active_sensor_idx, historical_window=historical_window, target_sensor_weight=target_sensor_weight)
    metrics = TrainingMetrics()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    
    for episode in range(num_episodes):
        total_reward = 0
        state = agent.get_state()
        
        for step in range(agent.steps_per_episode):  # max steps per episode
            initial_readings = agent.get_all_sensor_readings()
            actions = agent.act(state)
            
            # Apply actions to actuators
            for actuator, action_idx in zip(rob.get_actuators(), actions):
                motor_intensity = agent.action_space[action_idx]
                actuator.apply_intensity(motor_intensity * actuator.max_speed)
            
            # Run simulation steps
            for _ in range(100):
                rob.step()
            
            # Get new state and readings
            next_state = agent.get_state()
            final_readings = agent.get_all_sensor_readings()
            
            # Calculate reward and update metrics
            reward, metrics_data = agent.get_reward_with_metrics(initial_readings, final_readings)
            
            # Update metrics
            metrics.update(
                episode=episode,
                step=step,
                raw_reading=final_readings[active_sensor_idx],
                norm_reading=metrics_data['normalized_target'],
                stability=metrics_data['stability_penalty'],
                hist_diff=metrics_data['historical_diff'],
                raw_others={i: r for i, r in enumerate(final_readings) if i != active_sensor_idx},
                norm_others=metrics_data['normalized_others'],
                reward=reward,
                epsilon=agent.epsilon
            )
            
            # Store experience and train
            agent.remember(state, actions, reward, next_state)
            agent.replay()
            
            total_reward += reward
            state = next_state
            
            if step % 100 == 0:
                agent.update_target_network()
                print(f"Episode {episode + 1}, Step {step}",
                    f"  Raw Reading: {final_readings[active_sensor_idx]:.3f}", 
                    f"  Normalized Reading: {metrics_data['normalized_target']:.3f}",
                    f"  Stability Penalty: {metrics_data['stability_penalty']:.3f}",
                    f"  Historical Difference: {metrics_data['historical_diff']:.3f}",
                    f"  Reward: {reward:.3f}"
                )
        
        # End of episode
        metrics.end_episode(episode, total_reward)
        
        # Print episode summary
        print(f"\nEpisode {episode + 1}/{num_episodes} Complete")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Epsilon: {agent.epsilon:.2f}")
        print("-" * 50)
    
    # Save all metrics
    metrics.save_metrics(output_dir, timestamp)
    
    return agent, metrics

time_start = datetime.now().strftime("%Y-%m-%d-%H%M%S")

# output dir 
output_dir = 'output'
# Ensure directory exists
os.makedirs(output_dir, exist_ok=True)

historical_window = 10 # how many readings back are we subtracting from 
target_sensor_weight = 0.75 # ratio of how much we consider current sensor progress vs considering no change in other sensors 
num_episodes = 200
active_sensor_idx = 0  # Change this to use different sensors

run_output_dir = os.path.join(output_dir, f'{time_start}-{target_sensor_weight}-{historical_window}')
os.makedirs(run_output_dir, exist_ok=True)

# Train the robot
rob = Rob()
agent, metrics = train_robot(rob, active_sensor_idx=active_sensor_idx, num_episodes=num_episodes, target_sensor_weight=target_sensor_weight, historical_window=historical_window, output_dir=run_output_dir)

# save model 
save_model(agent, run_output_dir)

time_stop = datetime.now().strftime("%Y-%m-%d-%H%M%S")

print(f'Start time:{time_start}')
print(f'Stop time:{time_stop}')