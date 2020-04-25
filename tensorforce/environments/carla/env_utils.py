"""Utility functions for environment.py"""

import os
import cv2
import math
import random
import numpy as np
import carla
import pygame
import threading
import datetime

from typing import Union
from tensorforce.agents import Agent


# constants:
epsilon = np.finfo(np.float32).eps

# Use this dict to convert lanes objects to integers:
WAYPOINT_DICT = dict(lane_change={carla.LaneChange.NONE: 0,
                                  carla.LaneChange.Both: 1,
                                  carla.LaneChange.Left: 2,
                                  carla.LaneChange.Right: 3},
                     lane_type={carla.LaneType.NONE: 0,
                                carla.LaneType.Bidirectional: 1,
                                carla.LaneType.Biking: 2,
                                carla.LaneType.Border: 3,
                                carla.LaneType.Driving: 4,
                                carla.LaneType.Entry: 5,
                                carla.LaneType.Exit: 6,
                                carla.LaneType.Median: 7,
                                carla.LaneType.OffRamp: 8,
                                carla.LaneType.OnRamp: 9,
                                carla.LaneType.Parking: 10,
                                carla.LaneType.Rail: 11,
                                carla.LaneType.Restricted: 12,
                                carla.LaneType.RoadWorks: 13,
                                carla.LaneType.Shoulder: 14,
                                carla.LaneType.Sidewalk: 15,
                                carla.LaneType.Special1: 16,
                                carla.LaneType.Special2: 17,
                                carla.LaneType.Special3: 18,
                                carla.LaneType.Stop: 19,
                                carla.LaneType.Tram: 20,
                                carla.LaneType.Any: 21},
                     lane_marking_type={carla.LaneMarkingType.NONE: 0,
                                        carla.LaneMarkingType.BottsDots: 1,
                                        carla.LaneMarkingType.Broken: 2,
                                        carla.LaneMarkingType.BrokenBroken: 3,
                                        carla.LaneMarkingType.BrokenSolid: 4,
                                        carla.LaneMarkingType.Curb: 5,
                                        carla.LaneMarkingType.Grass: 6,
                                        carla.LaneMarkingType.Solid: 7,
                                        carla.LaneMarkingType.SolidBroken: 8,
                                        carla.LaneMarkingType.SolidSolid: 9,
                                        carla.LaneMarkingType.Other: 10},
                     traffic_light={carla.TrafficLightState.Green: 0,
                                    carla.TrafficLightState.Red: 1,
                                    carla.TrafficLightState.Yellow: 2,
                                    carla.TrafficLightState.Off: 3,
                                    carla.TrafficLightState.Unknown: 4}
                     )


# -------------------------------------------------------------------------------------------------
# -- PyGame
# -------------------------------------------------------------------------------------------------

def init_pygame():
    if not pygame.get_init():
        pygame.init()

    if not pygame.font.get_init():
        pygame.font.init()


def get_display(window_size, mode=pygame.HWSURFACE | pygame.DOUBLEBUF):
    """Returns a display used to render images and text.
        :param window_size: a tuple (width: int, height: int)
        :param mode: pygame rendering mode. Default: pygame.HWSURFACE | pygame.DOUBLEBUF
        :return: a pygame.display instance.
    """
    return pygame.display.set_mode(window_size, mode)


def get_font(size=14):
    return pygame.font.Font(pygame.font.get_default_font(), size)


def display_image(display, image, window_size=(800, 600), blend=False):
    """Displays the given image on a pygame window
    :param blend: whether to blend or not the given image.
    :param window_size: the size of the pygame's window. Default is (800, 600)
    :param display: pygame.display
    :param image: the image (numpy.array) to display/render on.
    """
    # Resize image if necessary
    if (image.shape[1], image.shape[0]) != window_size:
        image = resize(image, size=window_size)

    image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

    if blend:
        image_surface.set_alpha(100)

    display.blit(image_surface, (0, 0))


def display_text(display, font, text: [str], color=(255, 255, 255), origin=(0, 0), offset=(0, 2)):
    position = origin

    for line in text:
        if isinstance(line, dict):
            display.blit(font.render(line.get('text'), True, line.get('color', color)), position)
        else:
            display.blit(font.render(line, True, color), position)

        position = (position[0] + offset[0], position[1] + offset[1])


def pygame_save(display, path: str, name: str = None):
    if name is None:
        name = 'image-' + str(datetime.datetime.now()) + '.jpg'

    thread = threading.Thread(target=lambda: pygame.image.save(display, os.path.join(path, name)))
    thread.start()


# -------------------------------------------------------------------------------------------------
# -- CARLA
# -------------------------------------------------------------------------------------------------

def get_client(address, port, timeout=2.0) -> carla.Client:
    """Connects to the simulator.
        @:returns a carla.Client instance if the CARLA simulator accepts the connection.
    """
    client: carla.Client = carla.Client(address, port)
    client.set_timeout(timeout)
    return client


def random_blueprint(world: carla.World, actor_filter='vehicle.*', role_name='agent') -> carla.ActorBlueprint:
    """Retrieves a random blueprint.
        :param world: a carla.World instance.
        :param actor_filter: a string used to filter (select) blueprints. Default: 'vehicle.*'
        :param role_name: blueprint's role_name, Default: 'agent'.
        :return: a carla.ActorBlueprint instance.
    """
    blueprints = world.get_blueprint_library().filter(actor_filter)
    blueprint: carla.ActorBlueprint = random.choice(blueprints)
    blueprint.set_attribute('role_name', role_name)

    if blueprint.has_attribute('color'):
        color = random.choice(blueprint.get_attribute('color').recommended_values)
        blueprint.set_attribute('color', color)

    if blueprint.has_attribute('driver_id'):
        driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
        blueprint.set_attribute('driver_id', driver_id)

    if blueprint.has_attribute('is_invincible'):
        blueprint.set_attribute('is_invincible', 'true')

    # set the max speed
    if blueprint.has_attribute('speed'):
        float(blueprint.get_attribute('speed').recommended_values[1])
        float(blueprint.get_attribute('speed').recommended_values[2])
    else:
        print("No recommended values for 'speed' attribute")

    return blueprint


def random_spawn_point(world_map: carla.Map, different_from: carla.Location = None) -> carla.Transform:
    """Returns a random spawning location.
        :param world_map: a carla.Map instance obtained by calling world.get_map()
        :param different_from: ensures that the location of the random spawn point is different from the one specified here.
        :return: a carla.Transform instance.
    """
    available_spawn_points = world_map.get_spawn_points()

    if different_from is not None:
        while True:
            spawn_point = random.choice(available_spawn_points)

            if spawn_point.location != different_from:
                return spawn_point
    else:
        return random.choice(available_spawn_points)


def spawn_actor(world: carla.World, blueprint: carla.ActorBlueprint, spawn_point: carla.Transform,
                attach_to: carla.Actor = None, attachment_type=carla.AttachmentType.Rigid) -> carla.Actor:
    """Tries to spawn an actor in a CARLA simulator.
        :param world: a carla.World instance.
        :param blueprint: specifies which actor has to be spawned.
        :param spawn_point: where to spawn the actor. A transform specifies the location and rotation.
        :param attach_to: whether the spawned actor has to be attached (linked) to another one.
        :param attachment_type: the kind of the attachment. Can be 'Rigid' or 'SpringArm'.
        :return: a carla.Actor instance.
    """
    actor = world.try_spawn_actor(blueprint, spawn_point, attach_to, attachment_type)

    if actor is None:
        raise ValueError(f'Cannot spawn actor. Try changing the spawn_point ({spawn_point}) to something else.')

    return actor


def get_blueprint(world: carla.World, actor_id: str) -> carla.ActorBlueprint:
    return world.get_blueprint_library().find(actor_id)


def global_to_local(point: carla.Location, reference: Union[carla.Transform, carla.Location, carla.Rotation]):
    """Translates a 3D point from global to local coordinates using the current transformation as reference"""
    if isinstance(reference, carla.Transform):
        reference.transform(point)
    elif isinstance(reference, carla.Location):
        carla.Transform(reference, carla.Rotation()).transform(point)
    elif isinstance(reference, carla.Rotation):
        carla.Transform(carla.Location(), reference).transform(point)
    else:
        raise ValueError('Argument "reference" is none of carla.Transform or carla.Location or carla.Rotation!')


# -------------------------------------------------------------------------------------------------
# -- Other
# -------------------------------------------------------------------------------------------------

def resize(image, size: (int, int), interpolation=cv2.INTER_CUBIC):
    """Resize the given image.
        :param image: a numpy array with shape (height, width, channels).
        :param size: (width, height) to resize the image to.
        :param interpolation: Default: cv2.INTER_CUBIC.
        :return: the reshaped image.
    """
    return cv2.resize(image, dsize=size, interpolation=interpolation)


def scale(num, from_interval=(-1.0, +1.0), to_interval=(0.0, 7.0)) -> float:
    """Scales (interpolates) the given number to a given interval.
        :param num: a number
        :param from_interval: the interval the number is assumed to lie in.
        :param to_interval: the target interval.
        :return: the scaled/interpolated number.
    """
    x = np.interp(num, from_interval, to_interval)
    return float(round(x))


def save_agent(agent: Agent, agent_name: str, directory: str, separate_dir=True) -> str:
    if separate_dir:
        save_path = os.path.join(directory, agent_name)
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = directory

    checkpoint_path = agent.save(directory=save_path, filename=agent_name)
    return checkpoint_path


def get_record_path(base_dir: str, prefix='ep', pattern='-'):
    dirs = sorted(os.listdir(base_dir))
    count = 0

    if len(dirs) > 0:
        count = 1 + int(dirs[-1].split(pattern)[1])

    record_path = os.path.join(base_dir, f'{prefix}{pattern}{count}')
    os.mkdir(record_path)

    return record_path


def replace_nans(data: dict, nan=0.0, pos_inf=0.0, neg_inf=0.0):
    """In-place replacement of non-numerical values, i.e. NaNs and +/- infinity"""
    for key, value in data.items():
        if np.isnan(value).any() or np.isinf(value).any():
            data[key] = np.nan_to_num(value, nan=nan, posinf=pos_inf, neginf=neg_inf)

    return data


# -------------------------------------------------------------------------------------------------
# -- Debug
# -------------------------------------------------------------------------------------------------

class Colors(object):
    """Wraps some carla.Color instances."""
    red = carla.Color(255, 0, 0)
    green = carla.Color(0, 255, 0)
    blue = carla.Color(47, 210, 231)
    cyan = carla.Color(0, 255, 255)
    yellow = carla.Color(255, 255, 0)
    orange = carla.Color(255, 162, 0)
    white = carla.Color(255, 255, 255)
    black = carla.Color(0, 0, 0)


def draw_transform(debug, trans, col=Colors.red, lt=-1):
    yaw_in_rad = math.radians(trans.rotation.yaw)
    pitch_in_rad = math.radians(trans.rotation.pitch)

    p1 = carla.Location(x=trans.location.x + math.cos(pitch_in_rad) * math.cos(yaw_in_rad),
                        y=trans.location.y + math.cos(pitch_in_rad) * math.sin(yaw_in_rad),
                        z=trans.location.z + math.sin(pitch_in_rad))

    debug.draw_arrow(trans.location, p1, thickness=0.05, arrow_size=0.1, color=col, life_time=lt)


def draw_radar_measurement(debug_helper: carla.DebugHelper, data: carla.RadarMeasurement, velocity_range=7.5,
                           size=0.075, life_time=0.06):
    """Code adapted from carla/PythonAPI/examples/manual_control.py:
        - White: means static points.
        - Red: indicates points moving towards the object.
        - Blue: denoted points moving away.
    """
    radar_rotation = data.transform.rotation
    for detection in data:
        azimuth = math.degrees(detection.azimuth) + radar_rotation.yaw
        altitude = math.degrees(detection.altitude) + radar_rotation.pitch

        # move to local coordinates:
        forward_vec = carla.Vector3D(x=detection.depth - 0.25)
        global_to_local(forward_vec,
                        reference=carla.Rotation(pitch=altitude, yaw=azimuth, roll=radar_rotation.roll))

        # draw:
        debug_helper.draw_point(data.transform.location + forward_vec, size=size, life_time=life_time,
                                persistent_lines=False, color=carla.Color(255, 255, 255))


# -------------------------------------------------------------------------------------------------
# -- Math
# -------------------------------------------------------------------------------------------------

def l2_norm(location1, location2):
    """Computes the Euclidean distance between two carla.Location objects."""
    dx = location1.x - location2.x
    dy = location1.y - location2.y
    dz = location1.z - location2.z
    return math.sqrt(dx**2 + dy**2 + dz**2) + epsilon


def vector_norm(vec: carla.Vector3D) -> float:
    """Returns the norm/magnitude (a scalar) of the given 3D vector."""
    return math.sqrt(vec.x**2 + vec.y**2 + vec.z**2)


def speed(actor: carla.Actor) -> float:
    """Returns the speed of the given actor in km/h."""
    return 3.6 * vector_norm(actor.get_velocity())


def dot_product(a: carla.Vector3D, b: carla.Vector3D) -> float:
    return a.x * b.x + a.y * b.y + a.z * b.z


def cosine_similarity(a: carla.Vector3D, b: carla.Vector3D) -> float:
    """-1: opposite vectors (pointing in the opposite direction),
        0: orthogonal,
        1: exactly the same (pointing in the same direction)
    """
    return dot_product(a, b) / (vector_norm(a) * vector_norm(b))
