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
    Baseline which aggregates per-state baselines.
    """

    def __init__(self, baselines, scope='aggregated-baseline', summary_labels=()):
        """
        Aggregated baseline.

        Args:
            baselines: Dict of per-state baseline specification dicts
        """

        self.baselines = dict()
        for name in sorted(baselines):
            self.baselines[name] = Baseline.from_spec(
                spec=baselines[name],
                kwargs=dict(summary_labels=summary_labels))

        self.linear = Linear(size=1, bias=0.0, scope='prediction', summary_labels=summary_labels)

        super(AggregatedBaseline, self).__init__(scope, summary_labels)

    def tf_predict(self, states, internals, update):
        predictions = list()
        for name in sorted(states):
            prediction = self.baselines[name].predict(states=states[name], internals=internals, update=update)
            predictions.append(prediction)
        predictions = tf.stack(values=predictions, axis=1)
        prediction = self.linear.apply(x=predictions)
        return tf.squeeze(input=prediction, axis=1)

    def tf_regularization_loss(self):
        regularization_loss = super(AggregatedBaseline, self).tf_regularization_loss()
        if regularization_loss is None:
            losses = list()
        else:
            losses = [regularization_loss]

        for name in sorted(self.baselines):
            regularization_loss = self.baselines[name].regularization_loss()
            if regularization_loss is not None:
                losses.append(regularization_loss)

        regularization_loss = self.linear.regularization_loss()
        if regularization_loss is not None:
            losses.append(regularization_loss)

        if len(losses) > 0:
            return tf.add_n(inputs=losses)
        else:
            return None

    def get_variables(self, include_nontrainable=False):
        baseline_variables = super(AggregatedBaseline, self).get_variables(include_nontrainable=include_nontrainable)
        baselines_variables = [
            variable for name in sorted(self.baselines)
            for variable in self.baselines[name].get_variables(include_nontrainable=include_nontrainable)
        ]
        linear_variables = self.linear.get_variables(include_nontrainable=include_nontrainable)

        return baseline_variables + baselines_variables + linear_variables
