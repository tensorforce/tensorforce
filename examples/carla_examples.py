"""A collection of examples for CARLAEnvironment"""

import pygame

from tensorforce import Agent
from tensorforce.environments import CARLAEnvironment


def training_example(num_episodes: int, max_episode_timesteps: int):
    # Instantiate the environment (run the CARLA simulator before doing this!)
    env = CARLAEnvironment(debug=True)

    # Create your own agent (here is just an example)
    agent = Agent.create(agent='ppo',
                         environment=env,
                         max_episode_timesteps=max_episode_timesteps,
                         batch_size=1)

    # Training loop (you couldn't use a Runner instead)
    # `weights_dir` and `record_dir` are `None` to prevent saving and recording
    env.train(agent=agent, 
              num_episodes=num_episodes, max_episode_timesteps=max_episode_timesteps, 
              weights_dir=None, record_dir=None)  

    pygame.quit()


def custom_env_example(num_episodes: int, max_episode_timesteps: int):
    # import some libs
    import carla
    import numpy as np

    from tensorforce.environments.carla_environment import CARLAEnvironment, SensorSpecs, env_utils

    # Subclass `CARLAEnvironment` to customize it:
    class MyCARLAEnvironment(CARLAEnvironment):
        # Change actions space: (throttle, steer, brake, reverse)
        ACTIONS_SPEC = dict(type='float', shape=(4,), min_value=-1.0, max_value=1.0)
        DEFAULT_ACTIONS = np.array([0.0, 0.0, 0.0, 0.0])

        # Define your own mapping: actions -> carla.VehicleControl
        def actions_to_control(self, actions):
            self.control.throttle = float((actions[0] + 1) / 2.0)
            self.control.steer = float(actions[1])
            self.control.brake = float((actions[2] + 1) / 2.0)
            self.control.reverse = bool(actions[3] > 0)
            self.control.hand_brake = False

        # Define which sensors to use:
        def default_sensors(self) -> dict:
            sensors = super().default_sensors()

            # Substitute the default rgb camera with a semantic segmentation camera
            sensors['camera'] = SensorSpecs.segmentation_camera(position='front', attachment_type='Rigid',
                                                                image_size_x=self.window_size[0],
                                                                image_size_y=self.window_size[1],
                                                                sensor_tick=self.tick_time)
            # Add a radar sensor
            sensors['radar'] = SensorSpecs.radar(position='radar', sensor_tick=self.tick_time)
            return sensors

        # Define a default agent (only used if env.train(agent=None, ...))
        def default_agent(self, **kwargs) -> Agent:
            return Agent.create(agent='ppo',
                                environment=self,
                                max_episode_timesteps=kwargs.get('max_episode_timesteps'),
                                batch_size=1)

        # Define your own reward function:
        def reward(self, actions, time_cost=-2.0):
            speed = env_utils.speed(self.vehicle)
            speed_limit = self.vehicle.get_speed_limit()

            if speed <= speed_limit:
                speed_penalty = -1.0 if speed < speed_limit / 2 else 0.0
            else:
                speed_penalty = speed_limit - speed

            return time_cost - self.collision_penalty * 2.0 + speed_penalty

        def render(self, sensors_data: dict):
            super().render(sensors_data)
            env_utils.draw_radar_measurement(debug_helper=self.world.debug, data=sensors_data['radar'])

    # Training:
    env = MyCARLAEnvironment(debug=True)

    env.train(agent=None,  # pass None to use the default_agent
              num_episodes=num_episodes, max_episode_timesteps=max_episode_timesteps,
              weights_dir=None, record_dir=None)

    pygame.quit()
