# Copyright 2017 reinforce.io. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from tensorforce.contrib.remote_environment import RemoteEnvironment, MsgPackNumpyProtocol
from tensorforce.contrib.state_settable_environment import StateSettableEnvironment
from tensorforce import TensorForceError
from cached_property import cached_property
import re
import time


class UE4Environment(RemoteEnvironment, StateSettableEnvironment):
    """
    A special RemoteEnvironment for UE4 game connections.
    Communicates with the remote to receive information on the definitions of action- and observation spaces.
    Sends UE4 Action- and Axis-mappings as RL-actions and receives observations back defined by ducandu plugin Observer objects placed in the Game
    (these could be camera pixels or other observations, e.g. a x/y/z position of some game actor).
    """
    def __init__(self, host="localhost", port=6025, connect=True):
        """
        Args:
             host (str): The hostname to connect to.
             port (int): The port to connect to.
             connect (bool): Whether to connect already in this c'tor.
        """
        RemoteEnvironment.__init__(self, host, port)

        self.game_name = None  # remote env should send a name of the game upon connection
        self.action_space_desc = None
        self.observation_space_desc = None

        self.protocol = MsgPackNumpyProtocol()

        if connect:
            self.connect()

    def __str__(self):
        return "UE4Environment({}:{}{})".format(self.host, self.port, "[connected; {}]".format(self.game_name) if self.socket else "")

    def connect(self):
        RemoteEnvironment.connect(self)

        # get action- and state-specs from our game
        self.protocol.send({"cmd": "get_spec"}, self.socket)
        response = self.protocol.recv(self.socket)

        if "observation_space_desc" not in response or "action_space_desc" not in response:
            raise TensorForceError("ERROR in UE4Environment.connect: no observation- or action-space-desc sent by remote server!")

        # observers
        self.observation_space_desc = response["observation_space_desc"]
        # action-mappings
        self.action_space_desc = response["action_space_desc"]

        # invalidate our observation_space and action_space caches
        if "observation_space" in self.__dict__:
            del self.__dict__["observation_space"]
        if "action_space" in self.__dict__:
            del self.__dict__["action_space"]

    def seed(self, seed=None):
        if not seed:
            seed = time.time()
        # send command
        self.protocol.send({"cmd": "seed", "value": int(seed)}, self.socket)
        # wait for response
        response = self.protocol.recv(self.socket)
        if "status" not in response:
            raise RuntimeError("Message without field 'status' received!")
        elif response["status"] != "ok":
            raise RuntimeError("Message 'status' for seed command is not 'ok' ({})!".format(response["status"]))
        return seed

    def reset(self):
        """
        same as step (no kwargs to pass), but needs to block and return observation_dict
        - stores the received observation in self.last_observation
        """
        # send command
        self.protocol.send({"cmd": "reset"}, self.socket)
        # wait for response
        response = self.protocol.recv(self.socket)
        if "obs_dict" not in response:
            raise TensorForceError("Message without field 'obs_dict' received!")
        return response["obs_dict"]

    def set(self, setters, **kwargs):
        if "cmd" in kwargs:
            raise TensorForceError("Key 'cmd' must not be present in **kwargs to method `set`!")

        # forward kwargs to remote (only add command: set)
        message = kwargs
        message["cmd"] = "set"

        # sanity check setters
        # solve single tuple with prop-name and value -> should become a list (len=1) of this tuple
        if len(setters) >= 2 and not isinstance(setters[1], (list, tuple)):
            setters = list((setters,))
        for set_cmd in setters:
            if not re.match(r'\w+(:\w+)*', set_cmd[0]):
                raise TensorForceError("ERROR: property ({}) in setter-command does not match correct pattern!".format(set_cmd[0]))
            if len(set_cmd) == 3 and not isinstance(set_cmd[2], bool):
                raise TensorForceError("ERROR: 3rd item in setter-command must be of type bool ('is_relative' flag)!")
        message["setters"] = setters

        self.protocol.send(message, self.socket)
        # wait for response
        response = self.protocol.recv(self.socket)
        if "obs_dict" not in response:
            raise TensorForceError("Message without field 'obs_dict' received!")
        return response["obs_dict"]

    def execute(self, actions):
        """
        Executes a single step in the UE4 game. This step may be comprised of one or more actual game ticks for all of which the same given
        action- and axis-input is repeated.
        UE4 distinguishes between action-mappings, which are boolean actions (e.g. jump or dont-jump) and axis-mappings, which are continuous actions
        like MoveForward with values between -1.0 (run backwards) and 1.0 (run forwards), 0.0 would mean: stop.

        Args:
            actions: a dict with the following optional fields defines
                delta_time (float): The fake delta time to use for each single game tick.
                num_ticks (int): The number of ticks to be executed in this step (each tick will repeat the same given actions).
                action_mappings (Union[List[tuple],tuple]): A list of tuples of the shape: ([action-name], [True|False])
                axis_mappings (Union[List[tuple],tuple]): A list of tuples of the shape: ([axis-name], [float-value])
        Returns: The observation dict after all ticks have been completed.
        """
        delta_time = actions.get("delta_time", 1 / 60)
        num_ticks = actions.get("num_ticks", 4)
        action_mappings = actions.get("action_mappings", None)
        axis_mappings = actions.get("axis_mappings", None)

        # assert 1/600 <= delta_time < 1  # make sure our deltas are in some reasonable range
        # assert 1 <= num_ticks <= 20  # same for num_ticks
        # re-translate incoming action names into keyboard keys for the server
        try:
            if action_mappings is None:
                action_mappings = []
            else:
                action_mappings = self.translate_abstract_actions_to_keys(action_mappings)
            if axis_mappings is None:
                axis_mappings = []
            else:
                axis_mappings = self.translate_abstract_actions_to_keys(axis_mappings)
        except KeyError as e:
            raise TensorForceError("Action- or axis-mapping with name '{}' not defined in connected UE4 game!".format(e))

        # message = {"cmd": "step", 'delta_time': 0.33,
        #     'action': [{'name': 'X', 'pressed': True}, {'name': 'Y', 'pressed': False}],
        #     'axis': [{'name': 'Left', 'delta': 1}, {'name': 'Right', 'delta': 0}]
        # }
        message = dict(cmd="step", delta_time=delta_time, num_ticks=num_ticks, actions=action_mappings, axes=axis_mappings)
        self.protocol.send(message, self.socket)
        # wait for response (blocks)
        response = self.protocol.recv(self.socket)
        if "obs_dict" not in response:
            raise TensorForceError("Message without field 'obs_dict' received!")
        self.last_observation = response["obs_dict"]  # cache last observation

        return response["obs_dict"], response.get("is_terminal", False), response.get("reward", 0.0)

    @cached_property
    def states(self):
        observation_space = {}
        # derive observation space from observation_space_desc
        if self.observation_space_desc:
            for key, desc in self.observation_space_desc.items():
                type_ = desc["type"]
                space = None
                if type_ == "Bool":
                    space = dict(type="float", shape=())
                elif type_ == "IntBox":
                    space = dict(type="float", shape=desc.get("shape", ()), min_value=desc.get("min", None), max_value=desc.get("max", None))
                elif type_ == "Continuous":
                    space = dict(type="float", shape=desc.get("shape", ()), min_value=desc.get("min", None), max_value=desc.get("max", None))
                # TODO: Enums
                # elif type_ == "enum":
                #    space = spaces.Discrete(desc["len"])

                if space:
                    observation_space[key] = space

        return observation_space

    @cached_property
    def actions(self):
        action_space = {}
        # derive action space from action_space_desc
        if self.action_space_desc:
            for key, properties in self.action_space_desc.items():
                # UE4 action mapping -> bool
                if properties["type"] == "action":
                    action_space[key] = dict(type="int", num_actions=2)
                # UE4 axis mapping -> continuous (float)
                else:
                    action_space[key] = dict(type="float", shape=())

        return action_space

    def translate_abstract_actions_to_keys(self, abstract):
        """
        Translates a list of tuples ([pretty mapping], [value]) to a list of tuples ([some key], [translated value])
        each single item in abstract will undergo the following translation:

        Example1:
        we want: "MoveRight": 5.0
        possible keys for the action are: ("Right", 1.0), ("Left", -1.0)
        result: "Right": 5.0 * 1.0 = 5.0

        Example2:
        we want: "MoveRight": -0.5
        possible keys for the action are: ("Left", -1.0), ("Right", 1.0)
        result: "Left": -0.5 * -1.0 = 0.5 (same as "Right": -0.5)
        """

        # solve single tuple with name and value -> should become a list (len=1) of this tuple
        if len(abstract) >= 2 and not isinstance(abstract[1], (list, tuple)):
            abstract = list((abstract,))

        # now go through the list and translate each axis into an actual keyboard key (or mouse event/etc..)
        ret = []
        for a in abstract:
            # first_key = key-name (action mapping) OR tuple (key-name, scale) (axis mapping)
            # first_key = self.action_space_desc[bytes(a[0], encoding="utf-8")]["keys"][0]
            first_key = self.action_space_desc[a[0]]["keys"][0]
            # action mapping
            if isinstance(first_key, (bytes, str)):
                ret.append((first_key, a[1]))
            # axis mapping
            else:
                # ret.append((first_key[0].decode(), a[1] * first_key[1]))
                ret.append((first_key[0], a[1] * first_key[1]))

        return ret

