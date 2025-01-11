# drive_robot.py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from robot_env import RobotEnv
from datetime import datetime
import os
import pandas as pd

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class QNetwork(nn.Module):
    def __init__(self, state_size, actions_per_actuator, num_actuators, sensor_history_size):
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.actions_per_actuator = actions_per_actuator
        self.num_actuators = num_actuators
        self.sensor_history_size = sensor_history_size
        
        # Calculate total input size (actuator states + sensor history)
        total_input_size = state_size + sensor_history_size
        
        # Shared layers with increased size to handle additional input
        self.fc1 = nn.Linear(total_input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        
        # Separate output layers for each actuator
        self.output_layers = nn.ModuleList([
            nn.Linear(128, actions_per_actuator) for _ in range(num_actuators)
        ])
        
        # Initialize weights
        for layer in [self.fc1, self.fc2, self.fc3] + list(self.output_layers):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            
        # Move model to GPU if available
        self.to(device)
        
    def forward(self, x, sensor_history):
        if torch.isnan(x).any() or torch.isnan(sensor_history).any():
            print("NaN detected in input!")
            x = torch.nan_to_num(x, 0.0)
            sensor_history = torch.nan_to_num(sensor_history, 0.0)
        
        # Concatenate actuator state and sensor history
        combined_input = torch.cat([x, sensor_history], dim=1)
        combined_input = torch.clamp(combined_input, -10, 10)
        
        # Forward pass through shared layers
        x = torch.relu(self.fc1(combined_input))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        
        # Separate outputs for each actuator
        outputs = [layer(x) for layer in self.output_layers]
        return torch.stack(outputs, dim=1)

class RobotDQNAgent:
    def __init__(self, rob, state_size, actions_per_actuator, active_sensor_idx=0, 
                 historical_window=5, target_sensor_weight=0.5, sensor_history_size=10):
        self.rob = rob
        self.state_size = state_size
        self.num_actuators = len(rob.get_actuators())
        self.actions_per_actuator = actions_per_actuator
        self.active_sensor_idx = active_sensor_idx
        self.sensor_history_size = sensor_history_size
        
        # Initialize sensor history buffer for neural network input
        self.nn_sensor_history = deque(maxlen=sensor_history_size)
        # Fill history with zeros initially
        self.nn_sensor_history.extend([0.0] * sensor_history_size)
        
        # Validate sensor index
        num_sensors = len(rob.get_sensors())
        if not 0 <= active_sensor_idx < num_sensors:
            raise ValueError(f"Invalid active_sensor_idx {active_sensor_idx}. Must be between 0 and {num_sensors-1}")
        
        # DQN hyperparameters
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # Increased gamma to value future rewards more
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999999
        self.learning_rate = 0.001
        self.batch_size = 32
        self.max_grad_norm = 1.0

        # Reward structure
        self.target_sensor_weight = target_sensor_weight
        self.steps_per_episode = 1000
        
        # Track histories for reward calculation
        self.historical_window = historical_window
        self.reading_history = deque(maxlen=10)
        self.sensor_histories = {i: deque(maxlen=100) for i in range(num_sensors)}
        self.sensor_stats = {i: {'min': None, 'max': None, 'mean': None, 'std': None} 
                           for i in range(num_sensors)}
        self.baseline_reading = None
        self.potential_scale = 0.5
        
        # Initialize networks with sensor history size
        self.q_network = QNetwork(state_size, actions_per_actuator, self.num_actuators, sensor_history_size)
        self.target_network = QNetwork(state_size, actions_per_actuator, self.num_actuators, sensor_history_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.action_space = np.linspace(-1, 1, actions_per_actuator)
   

    def update_sensor_history(self, sensor_reading):
        """Update the neural network's sensor history buffer"""
        self.nn_sensor_history.append(sensor_reading)
    
    def get_sensor_history_tensor(self):
        """Convert sensor history to tensor for neural network input"""
        return torch.FloatTensor(list(self.nn_sensor_history))
  
    def act(self, state):
        """Choose an action using epsilon-greedy policy"""
        if random.random() < self.epsilon:
            # Random actions for each actuator
            actions = [random.randrange(self.actions_per_actuator) 
                      for _ in range(self.num_actuators)]
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                sensor_history_tensor = torch.FloatTensor(list(self.nn_sensor_history)).unsqueeze(0).to(device)
                action_values = self.q_network(state_tensor, sensor_history_tensor)
                # Select best action for each actuator
                actions = action_values.squeeze().cpu().argmax(dim=1).tolist()
        
        return actions
    
    def remember(self, state, sensor_history, actions, reward, next_state, next_sensor_history):
        """Store experience with multiple actions and sensor history"""
        self.memory.append((state, sensor_history, actions, reward, next_state, next_sensor_history))
    
    def replay(self):
        """Experience replay with multiple actions"""
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states, sensor_histories, actions, rewards, next_states, next_sensor_histories = zip(*minibatch)
        
        # Convert to tensors and move to GPU
        states = torch.FloatTensor(np.array(states)).to(device)
        sensor_histories = torch.FloatTensor(np.array(sensor_histories)).to(device)
        actions = torch.LongTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(np.array(next_states)).to(device)
        next_sensor_histories = torch.FloatTensor(np.array(next_sensor_histories)).to(device)
        
        if torch.isnan(states).any() or torch.isnan(sensor_histories).any():
            print("NaN detected in batch data!")
            return
        
        try:
            # Current Q values for each actuator
            current_q_all = self.q_network(states, sensor_histories)
            current_q = torch.gather(current_q_all, 2, actions.unsqueeze(2)).squeeze(2)
            
            # Next Q values
            with torch.no_grad():
                next_q_all = self.target_network(next_states, next_sensor_histories)
                next_q = next_q_all.max(2)[0]
            
            # Target Q values
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
    """Save the Q-network of the agent to the specified directory."""
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"{model_name}.pth")
    # Move model to CPU before saving
    model_state = {k: v.cpu() for k, v in agent.q_network.state_dict().items()}
    torch.save(model_state, model_path)
    print(f"Model saved to {model_path}")

# Rest of the code remains the same...
def train_robot(env, num_episodes=100, output_dir='output'):
    agent = RobotDQNAgent(rob=env.rob, state_size=env.num_actuators, 
                         actions_per_actuator=len(env.action_space), 
                         active_sensor_idx=env.active_sensor_idx, 
                         historical_window=env.historical_window, 
                         target_sensor_weight=env.target_sensor_weight,
                         sensor_history_size=10)  
    metrics = TrainingMetrics()
    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        for step in range(agent.steps_per_episode):
            # Get current sensor reading and update history
            current_sensor = env.get_all_sensor_readings()[env.active_sensor_idx]
            agent.update_sensor_history(current_sensor)
            
            # Get actions using current state and sensor history
            actions = agent.act(state)
            next_state, reward, done, metrics_data = env.step(actions)
            
            # Update sensor history with new reading
            next_sensor = env.get_all_sensor_readings()[env.active_sensor_idx]
            agent.update_sensor_history(next_sensor)
            
            # Store experience with sensor histories
            agent.remember(state, 
                         list(agent.nn_sensor_history),
                         actions, 
                         reward, 
                         next_state,
                         list(agent.nn_sensor_history))
            
            # Update metrics with raw reading instead of normalized
            metrics.update(
                episode=episode,
                step=step,
                raw_reading=metrics_data['raw_target'],  # Changed from normalized_target to raw_target
                norm_reading=metrics_data['normalized_target'],
                stability=metrics_data['stability_penalty'],
                hist_diff=metrics_data['historical_diff'],
                raw_others={i: metrics_data['raw_others'][i] for i in metrics_data['raw_others']},  # Changed to use raw_others
                norm_others=metrics_data['normalized_others'],
                reward=reward,
                epsilon=agent.epsilon
            )
            
            # Train the network
            agent.replay()
            
            total_reward += reward
            state = next_state
            
            if step % 100 == 0:
                agent.update_target_network()
                print(f"Episode {episode + 1}, Step {step}", 
                      f"  Raw Reading: {metrics_data['raw_target']:.3f}",  # Updated to show raw reading
                      f"  Stability Penalty: {metrics_data['stability_penalty']:.3f}",
                      f"  Historical Difference: {metrics_data['historical_diff']:.3f}",
                      f"  Reward: {reward:.3f}"
                )
        
        metrics.end_episode(episode, total_reward)
        print(f"\nEpisode {episode + 1}/{num_episodes} Complete")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Epsilon: {agent.epsilon:.2f}")
        print("-" * 50)
    
    metrics.save_metrics(output_dir, timestamp)
    return agent, metrics

def main():
    time_start = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    historical_window = 15 
    target_sensor_weight = 1.0
    num_episodes = 2000
    active_sensor_idx = 0  

    run_output_dir = os.path.join(output_dir, f'{time_start}-{target_sensor_weight}-{historical_window}')
    os.makedirs(run_output_dir, exist_ok=True)

    # Initialize environment
    env = RobotEnv(active_sensor_idx=active_sensor_idx, historical_window=historical_window, target_sensor_weight=target_sensor_weight)
    
    # Train the robot
    agent, _ = train_robot(env, num_episodes=num_episodes, output_dir=run_output_dir)

    # Save model 
    save_model(agent, run_output_dir)

    time_stop = datetime.now().strftime("%Y-%m-%d-%H%M%S")

    print(f'Start time:{time_start}')
    print(f'Stop time:{time_stop}')

if __name__ == "__main__":
    main()