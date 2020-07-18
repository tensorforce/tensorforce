import math
import numpy as np

from datetime import datetime

from tensorforce.environments import Environment

try:
    import carla
    import pygame

    from tensorforce.environments.carla import env_utils
    from tensorforce.environments.carla.env_utils import WAYPOINT_DICT
    from tensorforce.environments.carla.sensors import Sensor, SensorSpecs
    from tensorforce.environments.carla.synchronous_mode import CARLASyncContext
except ImportError:
    pass


class CARLAEnvironment(Environment):
    """A TensorForce Environment for the [CARLA driving simulator](https://github.com/carla-simulator/carla).
        - This environment is "synchronized" with the server, meaning that the server waits for a client tick. For a
          detailed explanation of this, please refer to https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/.
        - Subclass to customize the behaviour of states, actions, sensors, reward function, agent, training loop, etc.

       Requires, you to:
        - Install `pygame`, `opencv`
        - Install the CARLA simulator (version >= 0.9.8): https://carla.readthedocs.io/en/latest/start_quickstart
        - Install CARLA's Python bindings:
        --> Follow this [guide](https://carla.readthedocs.io/en/latest/build_system/#pythonapi), if you have trouble
            with that then follow the above steps.
        --> `cd your-path-to/CARLA_0.9.x/PythonAPI/carla/dist/`
        --> Extract `carla-0.9.x-py3.5-YOUR_OS-x86_64.egg` where `YOUR_OS` depends on your OS, i.e. `linux` or `windows`
        --> Create a `setup.py` file within the extracted folder and write the following:
          ```python
          from distutils.core import setup

          setup(name='carla',
                version='0.9.x',
                py_modules=['carla'])
          ```
        --> Install via pip: `pip install -e ~/CARLA_0.9.x/PythonAPI/carla/dist/carla-0.9.x-py3.5-YOUR_OS-x86_64`
        - Run the CARLA simulator from command line: `your-path-to/CARLA_0.9.x/./CarlaUE4.sh` or (CarlaUE4.exe)
        --> To use less resources add these flags: `-windowed -ResX=8 -ResY=8 --quality-level=Low`

        Hardware requirements (recommended):
        - GPU: dedicated, with at least 2/4 GB.
        - RAM: 16 GB suggested.
        - CPU: multicore, at least 4.
        - Note: on my hardware (i7 4700HQ 4C/8T, GT 750M 4GB, 16GB RAM) I achieve about 20 FPS in low quality mode.

        Example usage:
        - See [tensorforce/examples](https://github.com/tensorforce/tensorforce/tree/master/examples)
    
        Known Issues:
        - TensorForce's Runner is currently not compatible with this environment!

        Author:
        - Luca Anzalone (@luca96)
    """
    # States and actions specifications:
    # Actions: throttle or brake, steer, reverse (bool)
    ACTIONS_SPEC = dict(type='float', shape=(3,), min_value=-1.0, max_value=1.0)
    DEFAULT_ACTIONS = np.array([0.0, 0.0, 0.0])

    # Vehicle: speed, control (4), accelerometer (x, y, z), gyroscope (x, y, z), position (x, y), compass
    VEHICLE_FEATURES_SPEC = dict(type='float', shape=(14,))

    # Road: intersection (bool), junction (bool), speed_limit, lane_width, lane_change, left_lane, right_lane
    ROAD_FEATURES_SPEC = dict(type='float', shape=(8,))

    # TODO: add a loading map functionality (specified or at random) - load_map
    def __init__(self, address='localhost', port=2000, timeout=2.0, image_shape=(150, 200, 3), window_size=(800, 600),
                 vehicle_filter='vehicle.*', sensors: dict = None, route_resolution=2.0, fps=30.0, render=True,
                 debug=False):
        """
        :param address: CARLA simulator's id address. Required only if the simulator runs on a different machine.
        :param port: CARLA simulator's port.
        :param timeout: connection timeout.
        :param image_shape: shape of the images observations.
        :param window_size: pygame's window size. Meaningful only if `visualize=True`.
        :param vehicle_filter: use to spawn a particular vehicle (e.g. 'vehicle.tesla.model3') or class of vehicles
            (e.g. 'vehicle.audi.*')
        :param sensors: specifies which sensors should be equipped to the vehicle, better specified by subclassing
            `default_sensors()`.
        :param route_resolution: route planner resolution grain.
        :param fps: maximum framerate, it depends on your compiting power.
        :param render: if True a pygame window is shown.
        :param debug: enable to display some useful information about the vehicle.
        """
        super().__init__()
        env_utils.init_pygame()

        self.timeout = timeout
        self.client = env_utils.get_client(address, port, self.timeout)
        self.world = self.client.get_world()  # type: carla.World
        self.map = self.world.get_map()  # type: carla.Map
        self.synchronous_context = None

        # set fix fps:
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            synchronous_mode=False,
            fixed_delta_seconds=1.0 / fps))

        # vehicle
        self.vehicle_filter = vehicle_filter
        self.vehicle = None  # type: carla.Vehicle

        # actions
        self.control = None  # type: carla.VehicleControl
        self.prev_actions = None

        # weather
        # TODO: add weather support

        # visualization and debugging stuff
        self.image_shape = image_shape
        self.image_size = (image_shape[1], image_shape[0])
        self.DEFAULT_IMAGE = np.zeros(shape=self.image_shape, dtype=np.float32)
        self.fps = fps
        self.tick_time = 1.0 / self.fps
        self.should_render = render
        self.should_debug = debug
        self.clock = pygame.time.Clock()

        if self.should_render:
            self.window_size = window_size
            self.font = env_utils.get_font(size=13)
            self.display = env_utils.get_display(window_size)

        # variables for reward computation
        self.collision_penalty = 0.0

        # vehicle sensors suite
        self.sensors_spec = sensors if isinstance(sensors, dict) else self.default_sensors()
        self.sensors = dict()

    def states(self):
        return dict(image=dict(shape=self.image_shape),
                    vehicle_features=self.VEHICLE_FEATURES_SPEC,
                    road_features=self.ROAD_FEATURES_SPEC,
                    previous_actions=self.ACTIONS_SPEC)

    def actions(self):
        return self.ACTIONS_SPEC

    def reset(self, soft=False):
        self._reset_world(soft=soft)

        # reset actions
        self.control = carla.VehicleControl()
        self.prev_actions = self.DEFAULT_ACTIONS

        observation = env_utils.replace_nans(self._get_observation(sensors_data={}))
        return observation

    def reward(self, actions, time_cost=-1.0, a=2.0):
        """An example reward function. Subclass to define your own."""
        speed = env_utils.speed(self.vehicle)
        speed_limit = self.vehicle.get_speed_limit()

        if speed <= speed_limit:
            speed_penalty = 0.0
        else:
            speed_penalty = a * (speed_limit - speed)

        return time_cost - self.collision_penalty + speed_penalty

    def execute(self, actions, record_path: str = None):
        self.prev_actions = actions

        pygame.event.get()
        self.clock.tick()

        sensors_data = self.world_step(actions, record_path=record_path)

        reward = self.reward(actions)
        terminal = self.terminal_condition()
        next_state = env_utils.replace_nans(self._get_observation(sensors_data))

        self.collision_penalty = 0.0

        return next_state, terminal, reward

    def terminal_condition(self):
        """Tells whether the episode is terminated or not. Override with your own termination condition."""
        return False

    def close(self):
        super().close()

        if self.vehicle:
            self.vehicle.destroy()

        for sensor in self.sensors.values():
            sensor.destroy()

    def train(self, agent, num_episodes: int, max_episode_timesteps: int, weights_dir='weights/agents',
              agent_name='carla-agent', load_agent=False, record_dir='data/recordings', skip_frames=25):
        record_path = None
        should_record = isinstance(record_dir, str)
        should_save = isinstance(weights_dir, str)

        if agent is None:
            print(f'Using default agent...')
            agent = self.default_agent(max_episode_timesteps=max_episode_timesteps)

        try:
            if load_agent:
                agent.load(directory=os.path.join(weights_dir, agent_name), filename=agent_name, environment=self,
                           format='tensorflow')
                print('Agent loaded.')

            for episode in range(num_episodes):
                states = self.reset()
                total_reward = 0.0

                if should_record:
                    record_path = env_utils.get_record_path(base_dir=record_dir)
                    print(f'Recording in {record_path}.')

                with self.synchronous_context:
                    self.skip(num_frames=skip_frames)
                    t0 = datetime.now()

                    for i in range(max_episode_timesteps):
                        actions = agent.act(states)
                        states, terminal, reward = self.execute(actions, record_path=record_path)

                        total_reward += reward
                        terminal = terminal or (i == max_episode_timesteps - 1)

                        if agent.observe(reward, terminal):
                            print(f'{i + 1}/{max_episode_timesteps} -> update performed.')

                        if terminal:
                            elapsed = str(datetime.now() - t0).split('.')[0]
                            print(f'Episode-{episode} completed in {elapsed}, total_reward: {round(total_reward, 2)}\n')
                            break

                if should_save:
                    env_utils.save_agent(agent, agent_name, directory=weights_dir)
                    print('Agent saved.')
        finally:
            self.close()

    def default_sensors(self) -> dict:
        """Returns a predefined dict of sensors specifications"""
        return dict(imu=SensorSpecs.imu(),
                    collision=SensorSpecs.collision_detector(callback=self.on_collision),
                    camera=SensorSpecs.rgb_camera(position='top',
                                                  image_size_x=self.window_size[0], image_size_y=self.window_size[1],
                                                  sensor_tick=self.tick_time))

    def default_agent(self, **kwargs):
        """Returns a predefined agent for this environment"""
        raise NotImplementedError('Implement this to define your own default agent!')

    def on_collision(self, event, penalty=1000.0):
        impulse = math.sqrt(utils.vector_norm(event.normal_impulse))
        actor_type = event.other_actor.type_id

        if 'pedestrian' in actor_type:
            self.collision_penalty += penalty * impulse

        elif 'vehicle' in actor_type:
            self.collision_penalty += penalty / 2.0 * impulse
        else:
            self.collision_penalty += penalty * impulse

    def render(self, sensors_data: dict):
        """Renders sensors' output"""
        image = sensors_data['camera']
        env_utils.display_image(self.display, image, window_size=self.window_size)

    def debug(self, actions):
        env_utils.display_text(self.display, self.font, text=self.debug_text(actions), origin=(16, 12),
                               offset=(0, 16))

    def debug_text(self, actions):
        return ['%d FPS' % self.clock.get_fps(),
                '',
                'Throttle: %.2f' % self.control.throttle,
                'Steer: %.2f' % self.control.steer,
                'Brake: %.2f' % self.control.brake,
                'Reverse: %s' % ('T' if self.control.reverse else 'F'),
                'Hand brake: %s' % ('T' if self.control.hand_brake else 'F'),
                'Gear: %s' % {-1: 'R', 0: 'N'}.get(self.control.gear),
                '',
                'Speed %.1f km/h' % env_utils.speed(self.vehicle),
                'Speed limit %.1f km/h' % self.vehicle.get_speed_limit(),
                '',
                'Reward: %.2f' % self.reward(actions),
                'Collision penalty: %.2f' % self.collision_penalty]

    def skip(self, num_frames=10):
        """Skips the given amount of frames"""
        for _ in range(num_frames):
            self.synchronous_context.tick(timeout=self.timeout)

        if num_frames > 0:
            print(f'Skipped {num_frames} frames.')

    def before_world_step(self):
        """Callback: called before world.tick()"""
        pass

    def after_world_step(self, sensors_data: dict):
        """Callback: called after world.tick()."""
        pass

    def on_sensors_data(self, data: dict) -> dict:
        """Callback. Triggers when a world's 'tick' occurs, meaning that data from sensors are been collected because a
        simulation step of the CARLA's world has been completed.
            - Use this method to preprocess sensors' output data for: rendering, observation, ...
        """
        data['camera'] = self.sensors['camera'].convert_image(data['camera'])
        return data

    def world_step(self, actions, record_path: str = None):
        """Applies the actions to the vehicle, and updates the CARLA's world"""
        # [pre-tick updates] Apply control to update the vehicle
        self.actions_to_control(actions)
        self.vehicle.apply_control(self.control)

        self.before_world_step()

        # Advance the simulation and wait for sensors' data.
        data = self.synchronous_context.tick(timeout=self.timeout)
        data = self.on_sensors_data(data)

        # [post-tick updates] Update world-related stuff
        self.after_world_step(data)

        # Draw and debug:
        if self.should_render:
            self.render(sensors_data=data)

            if self.should_debug:
                self.debug(actions)

            pygame.display.flip()

            if isinstance(record_path, str):
                env_utils.pygame_save(self.display, record_path)

        return data

    def _reset_world(self, soft=False):
        # init actor
        if not soft:
            spawn_point = env_utils.random_spawn_point(self.map)
        else:
            spawn_point = self.spawn_point

        if self.vehicle is None:
            blueprint = env_utils.random_blueprint(self.world, actor_filter=self.vehicle_filter)
            self.vehicle = env_utils.spawn_actor(self.world, blueprint, spawn_point)  # type: carla.Vehicle

            self._create_sensors()
            self.synchronous_context = CARLASyncContext(self.world, self.sensors, fps=self.fps)
        else:
            self.vehicle.apply_control(carla.VehicleControl())
            self.vehicle.set_velocity(carla.Vector3D(x=0.0, y=0.0, z=0.0))
            self.vehicle.set_transform(spawn_point)

        # reset reward variables
        self.collision_penalty = 0.0

    def actions_to_control(self, actions):
        """Specifies the mapping between an actions vector and the vehicle's control."""
        # throttle and brake are mutual exclusive:
        self.control.throttle = float(actions[0]) if actions[0] > 0 else 0.0
        self.control.brake = float(-actions[0]) if actions[0] < 0 else 0.0

        # steering
        self.control.steer = float(actions[1])

        # reverse motion:
        self.control.reverse = bool(actions[2] > 0)

    def _get_observation(self, sensors_data: dict):
        image = sensors_data.get('camera', self.DEFAULT_IMAGE)

        if image.shape != self.image_shape:
            image = env_utils.resize(image, size=self.image_size)

        # Normalize image's pixels value to -1, +1
        observation = dict(image=(2 * image - 255.0) / 255.0,
                           vehicle_features=self._get_vehicle_features(),
                           road_features=self._get_road_features(),
                           previous_actions=self.prev_actions)
        return observation

    def _get_vehicle_features(self):
        t = self.vehicle.get_transform()
        control = self.vehicle.get_control()

        imu_sensor = self.sensors['imu']
        gyroscope = imu_sensor.gyroscope
        accelerometer = imu_sensor.accelerometer

        return [
            env_utils.speed(self.vehicle),
            control.gear,
            control.steer,
            control.throttle,
            control.brake,
            # Accelerometer:
            accelerometer[0],
            accelerometer[1],
            accelerometer[2],
            # Gyroscope:
            gyroscope[0],
            gyroscope[1],
            gyroscope[2],
            # Location
            t.location.x,
            t.location.y,
            # Compass:
            math.radians(imu_sensor.compass)]

    def _get_road_features(self):
        waypoint = self.map.get_waypoint(self.vehicle.get_location())
        speed_limit = self.vehicle.get_speed_limit()

        return [float(waypoint.is_intersection),
                float(waypoint.is_junction),
                waypoint.lane_width,
                math.log2(speed_limit),
                # Lane:
                WAYPOINT_DICT['lane_type'][waypoint.lane_type],
                WAYPOINT_DICT['lane_change'][waypoint.lane_change],
                WAYPOINT_DICT['lane_marking_type'][waypoint.left_lane_marking.type],
                WAYPOINT_DICT['lane_marking_type'][waypoint.right_lane_marking.type]]

    def _create_sensors(self):
        for name, args in self.sensors_spec.items():
            kwargs = args.copy()
            sensor = Sensor.create(sensor_type=kwargs.pop('type'), parent_actor=self.vehicle, **kwargs)

            if name == 'world':
                raise ValueError(f'Cannot name a sensor `world` because is reserved.')

            self.sensors[name] = sensor
