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

import tensorflow as tf

from tensorforce.core.networks import Linear
from tensorforce.core.baselines import Baseline


class AggregatedBaseline(Baseline):
    """
    Baseline which aggregates per-state baselines
    """

    def __init__(self, baselines, scope='aggregated-baseline', summary_labels=()):
        """
        Aggregated baseline

        Args:
            baselines: Dict of per-state baseline specification dicts
        """

        with tf.name_scope(name=scope):
            self.baselines = dict()
            for name, baseline_spec in baselines.items():
                with tf.name_scope(name=(name + '-baseline')):
                    self.baselines[name] = Baseline.from_spec(
                        spec=baseline_spec,
                        kwargs=dict(summary_labels=summary_labels)
                    )

            self.linear = Linear(size=1, bias=0.0, scope='prediction')

        super(AggregatedBaseline, self).__init__(scope, summary_labels)

    def tf_predict(self, states):
        predictions = list()
        for name, state in states.items():
            prediction = self.baselines[name].predict(states=state)
            predictions.append(prediction)
        predictions = tf.stack(values=predictions, axis=1)
        prediction = self.linear.apply(x=predictions)
        return tf.squeeze(input=prediction, axis=1)

    def get_variables(self, include_non_trainable=False):
        return super(AggregatedBaseline, self).get_variables(include_non_trainable=include_non_trainable) + \
            self.linear.get_variables(include_non_trainable=include_non_trainable) + \
            [variable for name in sorted(self.baselines) for variable in self.baselines[name].get_variables(
                include_non_trainable=include_non_trainable
            )]
