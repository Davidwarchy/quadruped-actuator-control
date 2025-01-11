# robot_desc.py
from controller import Robot
from enum import Enum

class JointType(Enum):
    HIP = "hip"
    KNEE = "knee"

class Sensor:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.device = None

    def get_reading(self):
        if self.type == "distance":
            return self.device.getValue()
        elif self.type == "yaw":
            return self.device.getRollPitchYaw()[2]  # Yaw is the third component
        elif self.type == "pitch":
            return self.device.getRollPitchYaw()[1]  # Pitch is the second component
        elif self.type == "position":
            return self.device.getValue()


class Actuator:
    def __init__(self, name, max_speed):
        self.name = name
        self.max_speed = max_speed
        self.device = None
        self.position_sensor = None
        self.state = 'x'

    def apply_intensity(self, intensity):
        clamped_intensity = max(min(intensity, self.max_speed), -self.max_speed)
        self.device.setVelocity(clamped_intensity)

    def get_position(self):
        return self.position_sensor.getValue() if self.position_sensor else None

class Rob:
    # Class constants for easy configuration
    MAX_SPEEDS = {
        JointType.HIP: 5.0,
        JointType.KNEE: 0.05
    }
    
    # Motor configuration template
    MOTOR_CONFIG = {
        "hip_motor_l0": JointType.HIP,
        "hip_motor_l1": JointType.HIP,
        "hip_motor_r0": JointType.HIP,
        "hip_motor_r1": JointType.HIP,
        "knee_motor_l0": JointType.KNEE,
        "knee_motor_l1": JointType.KNEE,
        "knee_motor_r0": JointType.KNEE,
        "knee_motor_r1": JointType.KNEE
    }
    
    # Sensor configuration
    SENSOR_CONFIG = [
        ("ds_0", "distance"),
        ("imu", "yaw"),
        ("imu", "pitch")  # Add pitch as a sensor
    ]

    def __init__(self):
        self.sensors = []
        self.actuators = []
        self.robot = None

    def setup(self):
        self.robot = Robot()
        self.TIME_STEP = int(self.robot.getBasicTimeStep())
        self._setup_sensors()
        self._setup_actuators()

    def _setup_sensors(self):
        self.sensors = [Sensor(name, type) for name, type in self.SENSOR_CONFIG]
        for sensor in self.sensors:
            sensor.device = self.robot.getDevice(sensor.name)
            sensor.device.enable(self.TIME_STEP)

    def _setup_actuators(self):
        self.actuators = [
            Actuator(
                name=motor_name,
                max_speed=self.MAX_SPEEDS[joint_type]
            )
            for motor_name, joint_type in self.MOTOR_CONFIG.items()
        ]
        
        for actuator in self.actuators:
            actuator.device = self.robot.getDevice(actuator.name)
            sensor_name = f"{actuator.name}_sensor"
            actuator.position_sensor = self.robot.getDevice(sensor_name)
            actuator.position_sensor.enable(self.TIME_STEP)
            actuator.device.setPosition(float('inf'))
            actuator.device.setVelocity(0.0)

    def get_sensors(self):
        return self.sensors

    def get_actuators(self):
        return self.actuators

    def step(self):
        return self.robot.step(self.TIME_STEP)
