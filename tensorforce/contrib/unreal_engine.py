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
import itertools
import logging


class UE4Environment(RemoteEnvironment, StateSettableEnvironment):
    """
    A special RemoteEnvironment for UE4 game connections.
    Communicates with the remote to receive information on the definitions of action- and observation spaces.
    Sends UE4 Action- and Axis-mappings as RL-actions and receives observations back defined by MLObserver
    objects placed in the Game
    (these could be camera pixels or other observations, e.g. a x/y/z position of some game actor).
    """
    def __init__(
        self,
        host="localhost",
        port=6025,
        connect=True,
        discretize_actions=False,
        delta_time=1/60,
        num_ticks=4
    ):
        """
        Args:
            host (str): The hostname to connect to.
            port (int): The port to connect to.
            connect (bool): Whether to connect already in this c'tor.
            discretize_actions (bool): Whether to treat axis-mappings defined in UE4 game as discrete actions.
                This would be necessary e.g. for agents that use q-networks where the output are q-values per discrete
                state-action pair.
            delta_time (float): The fake delta time to use for each single game tick.
            num_ticks (int): The number of ticks to be executed in a single act call (each tick will
                repeat the same given actions).
        """
        RemoteEnvironment.__init__(self, host, port)

        # RemoteEnvironment should send a name of the game upon connection.
        self.game_name = None
        self.action_space_desc = None
        self.observation_space_desc = None

        self.discretize_actions = discretize_actions
        self.discretized_actions = None
        self.delta_time = delta_time
        self.num_ticks = num_ticks

        # Our tcp messaging protocol to use (simple len-header + msgpack-numpy-body).
        self.protocol = MsgPackNumpyProtocol()

        if connect:
            self.connect()

    def __str__(self):
        return "UE4Environment({}:{}{})".format(self.host, self.port, "[connected; {}]".
                                                format(self.game_name) if self.socket else "")

    def connect(self, timeout=600):
        RemoteEnvironment.connect(self, timeout)

        # Get action- and state-specs from our game.
        self.protocol.send({"cmd": "get_spec"}, self.socket)
        response = self.protocol.recv(self.socket, "utf-8")

        # Game's name
        self.game_name = response.get("game_name")  # keep non-mandatory for now
        # Observers
        if "observation_space_desc" not in response:
            raise TensorForceError("Response to `get_spec` does not contain field `observation_space_desc`!")
        self.observation_space_desc = response["observation_space_desc"]
        # Action-mappings
        if "action_space_desc" not in response:
            raise TensorForceError("Response to `get_spec` does not contain field `action_space_desc`!")
        self.action_space_desc = response["action_space_desc"]

        if self.discretize_actions:
            self.discretize_action_space_desc()

        # Invalidate our states- and actions caches.
        if "states" in self.__dict__:
            del self.__dict__["states"]
        if "actions" in self.__dict__:
            del self.__dict__["actions"]

    def seed(self, seed=None):
        if not seed:
            seed = time.time()
        # Send command.
        self.protocol.send({"cmd": "seed", "value": int(seed)}, self.socket)
        # Wait for response.
        response = self.protocol.recv(self.socket, "utf-8")
        if "status" not in response:
            raise TensorForceError("Message without field 'status' received!")
        elif response["status"] != "ok":
            raise TensorForceError("Message 'status' for seed command is not 'ok' ({})!".format(response["status"]))
        return seed

    def reset(self):
        """
        same as step (no kwargs to pass), but needs to block and return observation_dict
        - stores the received observation in self.last_observation
        """
        # Send command.
        self.protocol.send({"cmd": "reset"}, self.socket)
        # Wait for response.
        response = self.protocol.recv(self.socket)
        # Extract observations.
        return self.extract_observation(response)

    def set_state(self, setters, **kwargs):
        if "cmd" in kwargs:
            raise TensorForceError("Key 'cmd' must not be present in **kwargs to method `set`!")

        # Forward kwargs to remote (only add command: set).
        message = kwargs
        message["cmd"] = "set"

        # Sanity check given setters.
        # Solve single tuple with prop-name and value -> should become a list (len=1) of this tuple.
        if len(setters) >= 2 and not isinstance(setters[1], (list, tuple)):
            setters = list((setters,))
        for set_cmd in setters:
            if not re.match(r'\w+(:\w+)*', set_cmd[0]):
                raise TensorForceError("ERROR: property ({}) in setter-command does not match correct pattern!".
                                       format(set_cmd[0]))
            if len(set_cmd) == 3 and not isinstance(set_cmd[2], bool):
                raise TensorForceError("ERROR: 3rd item in setter-command must be of type bool ('is_relative' flag)!")
        message["setters"] = setters

        self.protocol.send(message, self.socket)
        # Wait for response.
        response = self.protocol.recv(self.socket)
        return self.extract_observation(response)

    def execute(self, action):
        """
        Executes a single step in the UE4 game. This step may be comprised of one or more actual game ticks for all of
        which the same given
        action- and axis-inputs (or action number in case of discretized actions) are repeated.
        UE4 distinguishes between action-mappings, which are boolean actions (e.g. jump or dont-jump) and axis-mappings,
        which are continuous actions
        like MoveForward with values between -1.0 (run backwards) and 1.0 (run forwards), 0.0 would mean: stop.
        """
        action_mappings, axis_mappings = [], []

        # TODO: what if more than one actions are passed?

        # Discretized -> each action is an int
        if self.discretize_actions:
            # Pull record from discretized_actions, which will look like: [A, Right, SpaceBar].
            combination = self.discretized_actions[action]
            # Translate to {"axis_mappings": [('A', 1.0), (Right, 1.0)], "action_mappings": [(SpaceBar, True)]}
            for key, value in combination:
                # Action mapping (True or False).
                if isinstance(value, bool):
                    action_mappings.append((key, value))
                # Axis mapping: always use 1.0 as value as UE4 already multiplies with the correct scaling factor.
                else:
                    axis_mappings.append((key, value))
        # Non-discretized: Each action is a dict of action- and axis-mappings defined in UE4 game's input settings.
        # Re-translate Incoming action names into keyboard keys for the server.
        elif action:
            try:
                action_mappings, axis_mappings = self.translate_abstract_actions_to_keys(action)
            except KeyError as e:
                raise TensorForceError("Action- or axis-mapping with name '{}' not defined in connected UE4 game!".
                                       format(e))

        # message = {"cmd": "step", 'delta_time': 0.33,
        #     'actions': [('X', True), ('Y', False)],
        #     'axes': [('Left': 1.0), ('Up': -1.0)]
        # }
        message = dict(
            cmd="step",
            delta_time=self.delta_time,
            num_ticks=self.num_ticks,
            actions=action_mappings,
            axes=axis_mappings
        )
        self.protocol.send(message, self.socket)
        # Wait for response (blocks).
        response = self.protocol.recv(self.socket)
        r = response.pop(b"_reward", 0.0)
        is_terminal = response.pop(b"_is_terminal", False)

        obs = self.extract_observation(response)
        # Cache last observation
        self.last_observation = obs
        return obs, is_terminal, r

    @cached_property
    def states(self):
        observation_space = {}
        # Derive observation space from observation_space_desc.
        if self.observation_space_desc:
            for key, desc in self.observation_space_desc.items():
                type_ = desc["type"]
                if type_ == "Bool":
                    space = dict(type="float", shape=())
                elif type_ == "IntBox":
                    space = dict(
                        type="float",
                        shape=desc.get("shape", ()),
                        min_value=desc.get("min", None),
                        max_value=desc.get("max", None)
                    )
                elif type_ == "Continuous":
                    space = dict(
                        type="float",
                        shape=desc.get("shape", ()),
                        min_value=desc.get("min", None),
                         max_value=desc.get("max", None)
                    )
                # TODO: Enums
                else:
                    raise TensorForceError("Unsupported space type {} coming from Environment ("
                                           "observation_space_desc)!".format(type_))

                observation_space[key] = space
        # Simplest case: if only one observer -> use that one.
        if len(observation_space) == 1:
            observation_space = list(observation_space.values())[0]
        return observation_space

    @cached_property
    def actions(self):
        # Derive action space from action_space_desc.
        if not self.action_space_desc:
            return {}

        # Discretize all mappings. Pretend that each single mapping and combination thereof is its own discrete action.
        # E.g. MoveForward=Up(1.0)+Down(-1.0) MoveRight=Right(1.0)+Left(-1.0) -> UpRight, UpLeft, Right, Left, Up, Down,
        # DownRight, DownLeft, Idle
        if self.discretize_actions:
            return dict(type="int", num_actions=len(self.discretized_actions))
        # Leave each mapping as independent action, which may be continuous and can be combined with all other actions
        # in any way.
        else:
            action_space = {}
            for action_name, properties in self.action_space_desc.items():
                # UE4 action mapping -> bool
                if properties["type"] == "action":
                    action_space[action_name] = dict(type="int", num_actions=2)
                # UE4 axis mapping -> continuous (float) unless we have discretized axes
                else:
                    min_ = 0.0
                    max_ = 0.0
                    for mapping in properties["keys"]:
                        if mapping[1] > max_:
                            max_ = mapping[1]
                        if mapping[1] < min_:
                            min_ = mapping[1]
                    action_space[action_name] = dict(type="float", shape=(), min_value=min_, max_value=max_)
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

        # Solve single tuple with name and value -> should become a list (len=1) of this tuple.
        if len(abstract) >= 2 and not isinstance(abstract[1], (list, tuple)):
            abstract = list((abstract,))

        # Now go through the list and translate each axis into an actual keyboard key (or mouse event/etc..).
        actions, axes = [], []
        for a in abstract:
            # first_key = key-name (action mapping or discretized axis mapping) OR tuple (key-name, scale) (continuous
            # axis mapping)
            first_key = self.action_space_desc[a[0]]["keys"][0]
            # action mapping
            if isinstance(first_key, (bytes, str)):
                actions.append((first_key, a[1]))
            # axis mapping
            elif isinstance(first_key, tuple):
                axes.append((first_key[0], a[1] * first_key[1]))
            else:
                raise TensorForceError("action_space_desc contains unsupported type for key {}!".format(a[0]))

        return actions, axes

    def discretize_action_space_desc(self):
        """
        Creates a list of discrete action(-combinations) in case we want to learn with a discrete set of actions,
        but only have action-combinations (maybe even continuous) available from the env.
        E.g. the UE4 game has the following action/axis-mappings:

        ```javascript
        {
        'Fire':
            {'type': 'action', 'keys': ('SpaceBar',)},
        'MoveRight':
            {'type': 'axis', 'keys': (('Right', 1.0), ('Left', -1.0), ('A', -1.0), ('D', 1.0))},
        }
        ```

        -> this method will discretize them into the following 6 discrete actions:

        ```javascript
        [
        [(Right, 0.0),(SpaceBar, False)],
        [(Right, 0.0),(SpaceBar, True)]
        [(Right, -1.0),(SpaceBar, False)],
        [(Right, -1.0),(SpaceBar, True)],
        [(Right, 1.0),(SpaceBar, False)],
        [(Right, 1.0),(SpaceBar, True)],
        ]
        ```

        """
        # Put all unique_keys lists in one list and itertools.product that list.
        unique_list = []
        for nice, record in self.action_space_desc.items():
            list_for_record = []
            if record["type"] == "axis":
                # The main key for this record (always the first one)
                head_key = record["keys"][0][0]
                # The reference value (divide by this one to get the others)
                head_value = record["keys"][0][1]
                # The zero key (idle action; axis scale=0.0)
                list_for_record.append((head_key, 0.0))
                set_ = set()
                for key_and_scale in self.action_space_desc[nice]["keys"]:
                    # Build unique lists of mappings (each axis value should only be represented once).
                    if key_and_scale[1] not in set_:
                        list_for_record.append((head_key, key_and_scale[1] / head_value))
                        set_.add(key_and_scale[1])
            else:
                # Action-mapping
                list_for_record = [(record["keys"][0], False), (record["keys"][0], True)]
            unique_list.append(list_for_record)

        def so(in_):
            # in_ is List[Tuple[str,any]] -> sort by concat'd sequence of str(any's)
            st = ""
            for i in in_:
                st += str(i[1])
            return st

        # Then sort and get the entire list of all possible sorted meaningful key-combinations.
        combinations = list(itertools.product(*unique_list))
        combinations = list(map(lambda x: sorted(list(x), key=lambda y: y[0]), combinations))
        combinations = sorted(combinations, key=so)
        # Store that list as discretized_actions.
        self.discretized_actions = combinations

    @staticmethod
    def extract_observation(message):
        if b"obs_dict" not in message:
            raise TensorForceError("Message without field 'obs_dict' received!")

        ret = message[b"obs_dict"]
        # Only one observer -> use that one (no dict of dicts).
        if len(ret) == 1:
            ret = list(ret.values())[0]
        return ret

