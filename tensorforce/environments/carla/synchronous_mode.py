import carla
import queue


class CARLASyncContext(object):
    """
    Context manager to synchronize output from different sensors. Synchronous
    mode is enabled as long as we are inside this context

        with CARLASyncContext(world, sensors) as sync_mode:
            while True:
                data = sync_mode.tick(timeout=1.0)

    This code is based on https://github.com/carla-simulator/carla/blob/master/PythonAPI/examples/synchronous_mode.py
    """

    def __init__(self, world, sensors: dict, fps=30):
        self.world = world
        self.sensors = sensors
        self.frame = None
        self.delta_seconds = 1.0 / fps
        self._settings = None

        # Make a queue for each sensor and for world:
        self._queues = dict()
        self._add_queue('world', self.world.on_tick)

        for name, sensor in self.sensors.items():
            self._add_queue(name, sensor.add_callback)

    def __enter__(self):
        self._settings = self.world.get_settings()
        self.frame = self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=False,
            fixed_delta_seconds=self.delta_seconds,
            synchronous_mode=True))

        for sensor in self.sensors.values():
            sensor.start()

        return self

    def __exit__(self, *args, **kwargs):
        self.world.apply_settings(self._settings)

        for sensor in self.sensors.values():
            sensor.stop()

    def tick(self, timeout):
        self.frame = self.world.tick()

        data = dict()
        for name, q in self._queues.items():

            if name != 'world' and self.sensors[name].is_detector:
                # Detectors retrieve data only when triggered so have to not wait
                data[name] = self._get_detector_data(q)
            else:
                # Cameras + other are sensors that retrieve data at every simulation step
                data[name] = self._get_sensor_data(q, timeout)

        return data

    def _add_queue(self, name, register_event):
        """Registers an even on its own queue identified by name"""
        q = queue.Queue()
        register_event(q.put)
        self._queues[name] = q

    @staticmethod
    def _get_detector_data(sensor_queue: queue.Queue):
        """Retrieves data for detector, the call is non-blocking thus doesn't wait for available data."""
        data = []

        while not sensor_queue.empty():
            data.append(sensor_queue.get_nowait())

        return data

    def _get_sensor_data(self, sensor_queue: queue.Queue, timeout: float):
        """Retrieves data for sensors (i.e. camera and other) it blocks waiting until timeout is expired."""
        while True:
            data = sensor_queue.get(timeout=timeout)

            if data.frame == self.frame:
                return data
