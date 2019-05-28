# Copyright 2018 Tensorforce Team. All Rights Reserved.
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

from collections import OrderedDict

import tensorforce.core
from tensorforce.core.objectives import Objective


class Plus(Objective):
    """
    Additive combination of two objectives (specification key: `plus`).

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        objective1 (specification): First objective configuration
            (<span style="color:#C00000"><b>required</b></span>).
        objective2 (specification): Second objective configuration
            (<span style="color:#C00000"><b>required</b></span>).
        summary_labels ('all' | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
    """

    def __init__(self, name, objective1, objective2, summary_labels=None):
        super().__init__(name=name, summary_labels=summary_labels)

        self.objective1 = self.add_module(
            name='first-objective', module=objective1, modules=tensorforce.core.objective_modules
        )
        self.objective2 = self.add_module(
            name='second-objective', module=objective2, modules=tensorforce.core.objective_modules
        )

    def tf_loss_per_instance(
        self, policy, states, internals, auxiliaries, actions, reward, **kwargs
    ):
        kwargs1 = OrderedDict()
        kwargs2 = OrderedDict()
        for key, value in kwargs.items():
            assert len(value) == 2 and (value[0] is not None or value[1] is not None)
            if value[0] is not None:
                kwargs1[key] = value[0]
            if value[1] is not None:
                kwargs2[key] = value[1]

        loss1 = self.objective1.loss_per_instance(
            policy=policy, states=states, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, **kwargs1
        )

        loss2 = self.objective2.loss_per_instance(
            policy=policy, states=states, internals=internals, auxiliaries=auxiliaries,
            actions=actions, reward=reward, **kwargs2
        )

        return loss1 + loss2

    def optimizer_arguments(self, **kwargs):
        arguments = super().optimizer_arguments()
        arguments1 = self.objective1.optimizer_arguments(**kwargs)
        arguments2 = self.objective1.optimizer_arguments(**kwargs)
        for key, function in arguments1:
            if key in arguments2:

                def plus_function(states, internals, auxiliaries, actions, reward):
                    value1 = function(
                        states=states, internals=internals, auxiliaries=auxiliaries,
                        actions=actions, reward=reward
                    )
                    value2 = arguments2[key](
                        states=states, internals=internals, auxiliaries=auxiliaries,
                        actions=actions, reward=reward
                    )
                    return (value1, value2)

            else:

                def plus_function(states, internals, auxiliaries, actions, reward):
                    value1 = function(
                        states=states, internals=internals, auxiliaries=auxiliaries,
                        actions=actions, reward=reward
                    )
                    return (value1, None)

            arguments[key] = plus_function

        for key, function in arguments2:
            if key not in arguments1:

                def plus_function(states, internals, auxiliaries, actions, reward):
                    value2 = function(
                        states=states, internals=internals, auxiliaries=auxiliaries,
                        actions=actions, reward=reward
                    )
                    return (None, value2)

            arguments[key] = plus_function

        return arguments
