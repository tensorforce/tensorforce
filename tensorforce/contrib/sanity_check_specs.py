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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorforce import TensorForceError
import copy


def sanity_check_states(states_spec):
    """
    Sanity checks a states dict, used to define the state space for an MDP.
    Throws an error or warns if mismatches are found.

    Args:
        states_spec (Union[None,dict]): The spec-dict to check (or None).

    Returns: Tuple of 1) the state space desc and 2) whether there is only one component in the state space.
    """
    # Leave incoming states dict intact.
    states = copy.deepcopy(states_spec)

    # Unique state shortform.
    is_unique = ('shape' in states)
    if is_unique:
        states = dict(state=states)

    # Normalize states.
    for name, state in states.items():
        # Convert int to unary tuple.
        if isinstance(state['shape'], int):
            state['shape'] = (state['shape'],)

        # Set default type to float.
        if 'type' not in state:
            state['type'] = 'float'

    return states, is_unique


def sanity_check_actions(actions_spec):
    """
    Sanity checks an actions dict, used to define the action space for an MDP.
    Throws an error or warns if mismatches are found.

    Args:
        actions_spec (Union[None,dict]): The spec-dict to check (or None).

    Returns: Tuple of 1) the action space desc and 2) whether there is only one component in the action space.
    """
    # Leave incoming spec-dict intact.
    actions = copy.deepcopy(actions_spec)

    # Unique action shortform.
    is_unique = ('type' in actions)
    if is_unique:
        actions = dict(action=actions)

    # Normalize actions.
    for name, action in actions.items():
        # Set default type to int
        if 'type' not in action:
            action['type'] = 'int'

        # Check required values
        if action['type'] == 'int':
            if 'num_actions' not in action:
                raise TensorForceError("Action requires value 'num_actions' set!")
        elif action['type'] == 'float':
            if ('min_value' in action) != ('max_value' in action):
                raise TensorForceError("Action requires both values 'min_value' and 'max_value' set!")

        # Set default shape to empty tuple (single-int, discrete action space)
        if 'shape' not in action:
            action['shape'] = ()

        # Convert int to unary tuple
        if isinstance(action['shape'], int):
            action['shape'] = (action['shape'],)

    return actions, is_unique


def sanity_check_execution_spec(execution_spec):
    """
    Sanity checks a execution_spec dict, used to define execution logic (distributed vs single, shared memories, etc..)
    and distributed learning behavior of agents/models.
    Throws an error or warns if mismatches are found.

    Args:
        execution_spec (Union[None,dict]): The spec-dict to check (or None). Dict needs to have the following keys:
            - type: "single", "distributed"
            - distributed_spec: The distributed_spec dict with the following fields:
                - cluster_spec: TensorFlow ClusterSpec object (required).
                - job: The tf-job name.
                - task_index: integer (required).
                - protocol: communication protocol (default: none, i.e. 'grpc').
            - session_config: dict with options for a TensorFlow ConfigProto object (default: None).

    Returns: A cleaned-up (in-place) version of the given execution-spec.
    """

    # default spec: single mode
    def_ = dict(type="single",
                distributed_spec=None,
                session_config=None)

    if execution_spec is None:
        return def_

    assert isinstance(execution_spec, dict), "ERROR: execution-spec needs to be of type dict (but is of type {})!".\
        format(type(execution_spec).__name__)

    type_ = execution_spec.get("type")

    # TODO: Figure out what exactly we need for options and what types we should support.
    if type_ == "distributed":
        def_ = dict(job="ps", task_index=0, cluster_spec={
            "ps": ["localhost:22222"],
            "worker": ["localhost:22223"]
        })
        def_.update(execution_spec.get("distributed_spec", {}))
        execution_spec["distributed_spec"] = def_
        execution_spec["session_config"] = execution_spec.get("session_config")
        return execution_spec
    elif type_ == "multi-threaded":
        return execution_spec
    elif type_ == "single":
        return execution_spec

    if execution_spec.get('num_parallel') != None:
        assert type(execution_spec['num_parallel']) is int, "ERROR: num_parallel needs to be of type int but is of type {}!".format(type(execution_spec['num_parallel']).__name__)
        assert execution_spec['num_parallel'] > 0, "ERROR: num_parallel needs to be > 0 but is equal to {}".format(execution_spec['num_parallel'])
        return execution_spec

    raise TensorForceError("Unsupported execution type specified ({})!".format(type_))
