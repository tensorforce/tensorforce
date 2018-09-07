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

import tensorflow as tf
from tensorforce import util
import tensorforce.core.explorations


class Exploration(object):
    """
    Abstract exploration object.
    """

    def __init__(self, scope='exploration', summary_labels=None):
        self.summary_labels = set(summary_labels or ())

        self.variables = dict()

        def custom_getter(getter, name, registered=False, **kwargs):
            variable = getter(name=name, registered=True, **kwargs)
            if registered:
                pass
            elif name in self.variables:
                assert variable is self.variables[name]
            else:
                assert not kwargs['trainable']
                self.variables[name] = variable
            return variable

        self.explore = tf.make_template(
            name_=(scope + '/explore'),
            func_=self.tf_explore,
            custom_getter_=custom_getter
        )

    def tf_explore(self, episode, timestep, shape):
        """
        Creates exploration value, e.g. compute an epsilon for epsilon-greedy or sample normal  
        noise.
        """
        raise NotImplementedError

    def get_variables(self):
        """
        Returns exploration variables.

        Returns:
            List of variables.
        """
        return [self.variables[key] for key in sorted(self.variables)]

    @staticmethod
    def from_spec(spec):
        """
        Creates an exploration object from a specification dict.
        """
        exploration = util.get_object(
            obj=spec,
            predefined_objects=tensorforce.core.explorations.explorations
        )
        assert isinstance(exploration, Exploration)
        return exploration
