# robot_env.py
import numpy as np
from robot_desc import Rob
from collections import deque

class RobotEnv:
    def __init__(self, active_sensor_idx=0, historical_window=5, target_sensor_weight=0.5):
        self.rob = Rob()
        self.setup()
        self.num_actuators = len(self.rob.get_actuators())
        self.action_space = np.linspace(-1, 1, 2)  # Assuming 2 actions per actuator for simplicity
        self.active_sensor_idx = active_sensor_idx
        self.historical_window = historical_window
        self.target_sensor_weight = target_sensor_weight

        # Track histories for all sensors
        num_sensors = len(self.rob.get_sensors())
        self.reading_history = deque(maxlen=self.historical_window)  # For active sensor potential calculation
        self.sensor_histories = {i: deque(maxlen=100) for i in range(num_sensors)}
        self.sensor_stats = {i: {'min': None, 'max': None, 'mean': None, 'std': None} for i in range(num_sensors)}

    def setup(self):
        self.rob.setup()

    def reset(self):
        """Reset the environment and return initial state."""
        return self._get_state()

    def step(self, actions):
        """
        Perform one step in the environment given the action.

        :param actions: List of action indices for each actuator
        :return: next_state, reward, done, info
        """
        initial_readings = self._get_all_sensor_readings()
        
        # Apply actions to actuators
        for actuator, action_idx in zip(self.rob.get_actuators(), actions):
            motor_intensity = self.action_space[action_idx]
            actuator.apply_intensity(motor_intensity * actuator.max_speed)
        
        # Simulate robot movement
        for _ in range(100):  # Assuming small steps for stability
            self.rob.step()
        
        next_state = self._get_state()
        final_readings = self._get_all_sensor_readings()
        reward, metrics_data = self._calculate_reward(initial_readings, final_readings)
        done = False  # Adjust based on your termination condition if needed
        
        return next_state, reward, done, metrics_data

    def _get_state(self):
        """Get current state (actuator positions) with validation."""
        state = np.array([act.get_position() for act in self.rob.get_actuators()])
        state = np.clip(np.nan_to_num(state, 0.0), -10, 10)
        return state

    def _get_all_sensor_readings(self):
        """Get readings from all sensors."""
        readings = []
        for i, sensor in enumerate(self.rob.get_sensors()):
            reading = sensor.get_reading()
            if np.isnan(reading):
                print(f"Warning: NaN reading from sensor {i}")
                reading = 0.0
            readings.append(reading)
            self._update_sensor_stats(i, reading)
        return readings

    def _update_sensor_stats(self, sensor_idx, reading):
        """Update running statistics for sensor normalization."""
        history = self.sensor_histories[sensor_idx]
        history.append(reading)
        
        if len(history) >= 10:  # Wait for some history to accumulate
            stats = self.sensor_stats[sensor_idx]
            readings_array = np.array(history)
            stats['min'] = np.min(readings_array)
            stats['max'] = np.max(readings_array)
            stats['mean'] = np.mean(readings_array)
            stats['std'] = np.std(readings_array) if len(history) > 1 else 1.0

    def _normalize_reading(self, sensor_idx, reading):
        """Normalize sensor reading based on historical statistics."""
        stats = self.sensor_stats[sensor_idx]
        
        if stats['mean'] is None:
            stats['mean'] = reading
            stats['std'] = 1.0
            stats['min'] = reading
            stats['max'] = reading
        
        if stats['std'] > 0:
            normalized = (reading - stats['mean']) / stats['std']
        else:
            normalized = reading - stats['mean']
            
        return np.clip(normalized, -3, 3)

    def _calculate_reward(self, initial_readings, final_readings):
        """Calculate reward based on sensor readings."""
        # Get both raw and normalized readings for target sensor
        target_raw_initial = initial_readings[self.active_sensor_idx]
        target_raw_final = final_readings[self.active_sensor_idx]
        target_initial = self._normalize_reading(self.active_sensor_idx, target_raw_initial)
        target_final = self._normalize_reading(self.active_sensor_idx, target_raw_final)
        
        # Calculate the historical differences over multiple windows
        historical_diffs = []
        for i in range(min(len(self.reading_history), self.historical_window)):
            prev_reading = self.reading_history[-(i+1)]  # Reading from past windows
            diff = target_final - prev_reading
            weight = 1 / (i + 1)  # Diminishing weight for older windows
            historical_diffs.append(weight * diff)
        
        # Add the final difference (without weight) as well
        historical_diffs.append(target_final - target_initial)
        
        # Calculate stability penalty and store both raw and normalized readings
        stability_penalty = 0
        normalized_others = {}
        raw_others = {}
        num_other_sensors = len(initial_readings) - 1

        for i, (init_reading, final_reading) in enumerate(zip(initial_readings, final_readings)):
            if i != self.active_sensor_idx:
                init_norm = self._normalize_reading(i, init_reading)
                final_norm = self._normalize_reading(i, final_reading)
                normalized_others[i] = final_norm
                raw_others[i] = final_reading
                stability_penalty += abs(final_norm - init_norm)

        if num_other_sensors > 0:
            stability_penalty /= num_other_sensors
        
        reward = (self.target_sensor_weight * sum(historical_diffs) - 
                (1 - self.target_sensor_weight) * stability_penalty)

        self.reading_history.append(target_final)

        metrics_data = {
            'raw_target': target_raw_final,
            'normalized_target': target_final,
            'historical_diff': sum(historical_diffs),
            'stability_penalty': stability_penalty,
            'normalized_others': normalized_others,
            'raw_others': raw_others
        }
        
        return reward, metrics_data

    def get_state(self):
        return self._get_state()
    
    def get_all_sensor_readings(self):
        return self._get_all_sensor_readings()
