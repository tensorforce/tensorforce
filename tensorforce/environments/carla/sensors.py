"""A collection of sensors helpers."""

import math
import numpy as np
import carla


class Sensor(object):
    """Base class for sensor wrappers."""
    def __init__(self, parent_actor: carla.Actor, transform=carla.Transform(), attachment_type=None,
                 attributes: dict = None):
        self.parent = parent_actor
        self.world = self.parent.get_world()
        self.attributes = attributes or dict()
        self.event_callbacks = []

        # Look for callback(s)
        if 'callback' in self.attributes:
            self.event_callbacks.append(self.attributes.pop('callback'))

        elif 'callbacks' in self.attributes:
            for callback in self.attributes.pop('callbacks'):
                self.event_callbacks.append(callback)

        # detector-sensors retrieve data only when triggered (not at each tick!)
        self.sensor, self.is_detector = self._spawn(transform, attachment_type)

    @property
    def name(self) -> str:
        raise NotImplementedError

    def set_parent_actor(self, actor: carla.Actor):
        self.parent = actor

    def add_callback(self, callback):
        assert callback is not None
        self.event_callbacks.append(callback)

    def clear_callbacks(self):
        self.event_callbacks.clear()

    @staticmethod
    def create(sensor_type, **kwargs):
        if sensor_type == 'sensor.other.collision':
            return CollisionDetector(**kwargs)

        elif sensor_type == 'sensor.other.lane_invasion':
            return LaneInvasionSensor(**kwargs)

        elif sensor_type == 'sensor.other.gnss':
            return GnssSensor(**kwargs)

        elif sensor_type == 'sensor.other.imu':
            return IMUSensor(**kwargs)

        elif sensor_type == 'sensor.camera.rgb':
            return RGBCameraSensor(**kwargs)

        elif sensor_type == 'sensor.camera.semantic_segmentation':
            return SemanticCameraSensor(**kwargs)

        elif sensor_type == 'sensor.camera.depth':
            return DepthCameraSensor(**kwargs)

        elif sensor_type == 'sensor.other.obstacle':
            return ObstacleDetector(**kwargs)

        elif sensor_type == 'sensor.lidar.ray_cast':
            return LidarSensor(**kwargs)

        elif sensor_type == 'sensor.other.radar':
            return RadarSensor(**kwargs)
        else:
            raise ValueError(f'String `{sensor_type}` does not denote a valid sensor!')

    def start(self):
        """Start listening for events"""
        if not self.sensor.is_listening:
            self.sensor.listen(self.on_event)
        else:
            print(f'Sensor {self.name} is already been started!')

    def stop(self):
        """Stop listening for events"""
        self.sensor.stop()

    def _spawn(self, transform, attachment_type=None):
        """Spawns itself within a carla.World."""
        if attachment_type is None:
            attachment_type = carla.AttachmentType.Rigid

        sensor_bp: carla.ActorBlueprint = self.world.get_blueprint_library().find(self.name)

        for attr, value in self.attributes.items():
            if sensor_bp.has_attribute(attr):
                sensor_bp.set_attribute(attr, str(value))
            else:
                print(f'Sensor {self.name} has no attribute `{attr}`')

        sensor_actor = self.world.spawn_actor(sensor_bp, transform, self.parent, attachment_type)
        is_detector = not sensor_bp.has_attribute('sensor_tick')

        return sensor_actor, is_detector

    def on_event(self, event):
        for callback in self.event_callbacks:
            callback(event)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None

        self.parent = None
        self.world = None


# -------------------------------------------------------------------------------------------------
# -- Camera Sensors
# -------------------------------------------------------------------------------------------------

class CameraSensor(Sensor):
    def __init__(self, color_converter=carla.ColorConverter.Raw, **kwargs):
        super().__init__(**kwargs)
        self.color_converter = color_converter

    @property
    def name(self):
        raise NotImplementedError

    def convert_image(self, image: carla.Image, dtype=np.dtype("uint8")):
        if self.color_converter is not carla.ColorConverter.Raw:
            image.convert(self.color_converter)

        array = np.frombuffer(image.raw_data, dtype=dtype)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array


class RGBCameraSensor(CameraSensor):
    @property
    def name(self):
        return 'sensor.camera.rgb'


class DepthCameraSensor(CameraSensor):
    @property
    def name(self):
        return 'sensor.camera.depth'


class SemanticCameraSensor(CameraSensor):
    @property
    def name(self):
        return 'sensor.camera.semantic_segmentation'


# -------------------------------------------------------------------------------------------------
# -- Detector Sensors
# -------------------------------------------------------------------------------------------------

class CollisionDetector(Sensor):
    def __init__(self, parent_actor, **kwargs):
        super().__init__(parent_actor, **kwargs)

    @property
    def name(self):
        return 'sensor.other.collision'


class LaneInvasionSensor(Sensor):
    def __init__(self, parent_actor, **kwargs):
        super().__init__(parent_actor, **kwargs)

    @property
    def name(self):
        return 'sensor.other.lane_invasion'


class ObstacleDetector(Sensor):
    def __init__(self, parent_actor, **kwargs):
        super().__init__(parent_actor, **kwargs)

    @property
    def name(self):
        return 'sensor.other.obstacle'


# -------------------------------------------------------------------------------------------------
# -- Other Sensors
# -------------------------------------------------------------------------------------------------

class LidarSensor(Sensor):
    def __init__(self, parent_actor, **kwargs):
        super().__init__(parent_actor, **kwargs)

    @property
    def name(self):
        return 'sensor.lidar.ray_cast'


class RadarSensor(Sensor):
    def __init__(self, parent_actor, **kwargs):
        super().__init__(parent_actor, **kwargs)

    @property
    def name(self):
        return 'sensor.other.radar'

    @staticmethod
    def convert(radar_measurement: carla.RadarMeasurement):
        """Converts a carla.RadarMeasurement into a numpy array [[velocity, altitude, azimuth, depth]]"""
        points = np.frombuffer(radar_measurement.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (len(radar_measurement), 4))
        return points


class GnssSensor(Sensor):
    def __init__(self, parent_actor, transform=carla.Transform(carla.Location(x=1.0, z=2.8)), **kwargs):
        super().__init__(parent_actor, transform=transform, **kwargs)
        self.lat = 0.0
        self.lon = 0.0

    @property
    def name(self):
        return 'sensor.other.gnss'

    def on_event(self, event):
        super().on_event(event)
        self.lat = event.latitude
        self.lon = event.longitude

    def destroy(self):
        super().destroy()
        self.lat = None
        self.lon = None


class IMUSensor(Sensor):
    def __init__(self, parent_actor, **kwargs):
        super().__init__(parent_actor, **kwargs)
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0

    @property
    def name(self):
        return 'sensor.other.imu'

    def on_event(self, event):
        super().on_event(event)
        limits = (-99.9, 99.9)

        self.accelerometer = (
            max(limits[0], min(limits[1], event.accelerometer.x)),
            max(limits[0], min(limits[1], event.accelerometer.y)),
            max(limits[0], min(limits[1], event.accelerometer.z)))

        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(event.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(event.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(event.gyroscope.z))))

        self.compass = math.degrees(event.compass)

    def destroy(self):
        super().destroy()
        self.accelerometer = None
        self.gyroscope = None
        self.compass = None


# -------------------------------------------------------------------------------------------------
# -- Sensors specifications
# -------------------------------------------------------------------------------------------------

class SensorSpecs(object):
    ATTACHMENT_TYPE = {'SpringArm': carla.AttachmentType.SpringArm,
                       'Rigid': carla.AttachmentType.Rigid,
                       None: carla.AttachmentType.Rigid}

    COLOR_CONVERTER = {'Raw': carla.ColorConverter.Raw,
                       'CityScapesPalette': carla.ColorConverter.CityScapesPalette,
                       'Depth': carla.ColorConverter.Depth,
                       'LogarithmicDepth': carla.ColorConverter.LogarithmicDepth,
                       None: carla.ColorConverter.Raw}

    @staticmethod
    def get_position(position: str = None) -> carla.Transform:
        if position == 'top':
            return carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0))
        elif position == 'top-view':
            return carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0))
        elif position == 'front':
            return carla.Transform(carla.Location(x=1.5, z=1.8))
        elif position == 'on-top':
            return carla.Transform(carla.Location(x=-0.9, y=0.0, z=2.2))
        elif position == 'on-top2':
            return carla.Transform(carla.Location(x=0.0, y=0.0, z=2.2))
        else:
            return carla.Transform()

    @staticmethod
    def set(sensor_spec: dict, **kwargs):
        for key, value in kwargs.items():
            if key == 'position':
                sensor_spec['transform'] = SensorSpecs.get_position(value)
            elif key == 'attachment_type':
                sensor_spec[key] = SensorSpecs.ATTACHMENT_TYPE[value]
            elif key == 'color_converter':
                sensor_spec[key] = SensorSpecs.COLOR_CONVERTER[value]

    @staticmethod
    def set_color_converter(camera_spec: dict, color_converter: str = None):
        camera_spec['color_converter'] = SensorSpecs.COLOR_CONVERTER[color_converter]
        return SensorSpecs

    @staticmethod
    def camera(kind: str, transform: carla.Transform = None, position: str = None, attachment_type=None,
               color_converter=None, **kwargs) -> dict:
        assert kind in ['rgb', 'depth', 'semantic_segmentation']
        return dict(type='sensor.camera.' + kind,
                    transform=transform or SensorSpecs.get_position(position),
                    attachment_type=SensorSpecs.ATTACHMENT_TYPE[attachment_type],
                    color_converter=SensorSpecs.COLOR_CONVERTER[color_converter],
                    attributes=kwargs)

    @staticmethod
    def rgb_camera(transform: carla.Transform = None, position: str = None, attachment_type='SpringArm',
                   color_converter='Raw', **kwargs):
        return SensorSpecs.camera('rgb', transform, position, attachment_type, color_converter, **kwargs)

    @staticmethod
    def depth_camera(transform: carla.Transform = None, position: str = None, attachment_type='SpringArm',
                     color_converter='LogarithmicDepth', **kwargs):
        return SensorSpecs.camera('depth', transform, position, attachment_type, color_converter, **kwargs)

    @staticmethod
    def segmentation_camera(transform: carla.Transform = None, position: str = None, attachment_type='SpringArm',
                            color_converter='CityScapesPalette', **kwargs):
        return SensorSpecs.camera('semantic_segmentation', transform, position, attachment_type, color_converter, **kwargs)

    @staticmethod
    def detector(kind: str, transform: carla.Transform = None, position: str = None, attachment_type=None,
                 **kwargs) -> dict:
        assert kind in ['collision', 'lane_invasion', 'obstacle']
        return dict(type='sensor.other.' + kind,
                    transform=transform or SensorSpecs.get_position(position),
                    attachment_type=SensorSpecs.ATTACHMENT_TYPE[attachment_type],
                    attributes=kwargs)

    @staticmethod
    def collision_detector(transform: carla.Transform = None, position: str = None, attachment_type='Rigid', **kwargs):
        return SensorSpecs.detector('collision', transform, position, attachment_type, **kwargs)

    @staticmethod
    def lane_detector(transform: carla.Transform = None, position: str = None, attachment_type='Rigid', **kwargs):
        return SensorSpecs.detector('lane_invasion', transform, position, attachment_type, **kwargs)

    @staticmethod
    def obstacle_detector(transform: carla.Transform = None, position: str = None, attachment_type='Rigid', **kwargs):
        return SensorSpecs.detector('obstacle', transform, position, attachment_type, **kwargs)

    @staticmethod
    def other(kind: str, transform: carla.Transform = None, position: str = None, attachment_type=None, **kwargs) -> dict:
        assert kind in ['imu', 'gnss', 'radar']
        return dict(type='sensor.other.' + kind,
                    transform=transform or SensorSpecs.get_position(position),
                    attachment_type=SensorSpecs.ATTACHMENT_TYPE[attachment_type],
                    attributes=kwargs)

    @staticmethod
    def lidar(transform: carla.Transform = None, position: str = None, attachment_type=None, **kwargs) -> dict:
        return dict(type='sensor.lidar.ray_cast',
                    transform=transform or SensorSpecs.get_position(position),
                    attachment_type=SensorSpecs.ATTACHMENT_TYPE[attachment_type],
                    attributes=kwargs)

    @staticmethod
    def radar(transform: carla.Transform = None, position: str = None, attachment_type='Rigid', **kwargs):
        return SensorSpecs.other('radar', transform, position, attachment_type, **kwargs)

    @staticmethod
    def imu(transform: carla.Transform = None, position: str = None, attachment_type='Rigid', **kwargs):
        return SensorSpecs.other('imu', transform, position, attachment_type, **kwargs)

    @staticmethod
    def gnss(transform: carla.Transform = None, position: str = None, attachment_type='Rigid', **kwargs):
        return SensorSpecs.other('imu', transform, position, attachment_type, **kwargs)
