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

import inspect

from tensorforce.agents.agent import Agent
from tensorforce.meta_parameter_recorder import MetaParameterRecorder
from tensorforce.contrib.sanity_check_specs import sanity_check_execution_spec


class LearningAgent(Agent):
    """
    Base class for learning agents, using as model a subclass of MemoryModel and DistributionModel.
    """

    def __init__(
        self,
        states,
        actions,
        network,
        update_mode,
        memory,
        optimizer,
        batched_observe=True,
        batching_capacity=1000,
        scope='learning-agent',
        device=None,
        saver=None,
        summarizer=None,
        execution=None,
        variable_noise=None,
        states_preprocessing=None,
        actions_exploration=None,
        reward_preprocessing=None,
        discount=0.99,
        distributions=None,
        entropy_regularization=None
    ):
        """
        Initializes the learning agent.

        Args:
            update_mode (spec): Update mode specification, with the following attributes
                (required):
                - unit: one of 'timesteps', 'episodes', 'sequences' (required).
                - batch_size: integer (required).
                - frequency: integer (default: batch_size).
                - length: integer (optional if unit == 'sequences', default: 8).
            memory (spec): Memory specification, see core.memories module for more information
                (required).
            optimizer (spec): Optimizer specification, see core.optimizers module for more
                information (required).
            network (spec): Network specification, usually a list of layer specifications, see
                core.networks module for more information (required).
            scope (str): TensorFlow scope (default: name of agent).
            device: TensorFlow device (default: none)
            saver (spec): Saver specification, with the following attributes (default: none):
                - directory: model directory.
                - file: model filename (optional).
                - seconds or steps: save frequency (default: 600 seconds).
                - load: specifies whether model is loaded, if existent (default: true).
                - basename: optional file basename (default: 'model.ckpt').
            summarizer (spec): Summarizer specification, with the following attributes (default:
                none):
                - directory: summaries directory.
                - seconds or steps: summarize frequency (default: 120 seconds).
                - labels: list of summary labels to record (default: []).
                - meta_param_recorder_class: ???.
            execution (spec): Execution specification (see sanity_check_execution_spec for details).
            variable_noise (float): Standard deviation of variable noise (default: none).
            states_preprocessing (spec, or dict of specs): States preprocessing specification, see
                core.preprocessors module for more information (default: none)
            actions_exploration (spec, or dict of specs): Actions exploration specification, see
                core.explorations module for more information (default: none).
            reward_preprocessing (spec): Reward preprocessing specification, see core.preprocessors
                module for more information (default: none).
            discount (float): Discount factor for future rewards (default: 0.99).
            distributions (spec / dict of specs): Distributions specifications, see
                core.distributions module for more information (default: none).
            entropy_regularization (float): Entropy regularization weight (default: none).
        """

        self.scope = scope
        self.device = device
        self.saver = saver
        self.summarizer = summarizer
        self.execution = sanity_check_execution_spec(execution)
        self.variable_noise = variable_noise
        self.states_preprocessing = states_preprocessing
        self.actions_exploration = actions_exploration
        self.reward_preprocessing = reward_preprocessing
        self.update_mode = update_mode
        self.memory = memory
        self.optimizer = optimizer
        self.discount = discount
        self.network = network
        self.distributions = distributions
        self.entropy_regularization = entropy_regularization

        # TensorFlow summarizer & Configuration Meta Parameter Recorder options
        if self.summarizer is None:
            summary_labels = set()
        else:
            summary_labels = set(self.summarizer.get('labels', ()))

        self.meta_param_recorder = None

        # if 'configuration' in self.summary_labels or 'print_configuration' in self.summary_labels:
        if any(k in summary_labels for k in ['configuration', 'print_configuration']):
            self.meta_param_recorder = MetaParameterRecorder(inspect.currentframe())
            if 'meta_dict' in self.summarizer:
                # Custom Meta Dictionary passed
                self.meta_param_recorder.merge_custom(self.summarizer['meta_dict'])
            if 'configuration' in summary_labels:
                # Setup for TensorBoard population
                self.summarizer['meta_param_recorder_class'] = self.meta_param_recorder
            if 'print_configuration' in summary_labels:
                # Print to STDOUT (TODO: optimize output)
                self.meta_param_recorder.text_output(format_type=1)

        super(LearningAgent, self).__init__(
            states=states,
            actions=actions,
            batched_observe=batched_observe,
            batching_capacity=batching_capacity
        )

    def import_experience(self, experiences):
        """
        Imports experiences.

        Args:
            experiences: 
        """
        if isinstance(experiences, dict):
            if self.unique_state:
                experiences['states'] = dict(state=experiences['states'])
            if self.unique_action:
                experiences['actions'] = dict(action=experiences['actions'])

            self.model.import_experience(**experiences)

        else:
            if self.unique_state:
                states = dict(state=list())
            else:
                states = {name: list() for name in experiences[0]['states']}
            internals = [list() for _ in experiences[0]['internals']]
            if self.unique_action:
                actions = dict(action=list())
            else:
                actions = {name: list() for name in experiences[0]['actions']}
            terminal = list()
            reward = list()

            for experience in experiences:
                if self.unique_state:
                    states['state'].append(experience['states'])
                else:
                    for name in sorted(states):
                        states[name].append(experience['states'][name])
                for n, internal in enumerate(internals):
                    internal.append(experience['internals'][n])
                if self.unique_action:
                    actions['action'].append(experience['actions'])
                else:
                    for name in sorted(actions):
                        actions[name].append(experience['actions'][name])
                terminal.append(experience['terminal'])
                reward.append(experience['reward'])

            self.model.import_experience(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward
            )
