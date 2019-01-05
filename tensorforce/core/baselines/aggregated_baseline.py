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

import tensorflow as tf

from tensorforce.core import layer_modules
from tensorforce.core.baselines import Baseline


class AggregatedBaseline(Baseline):
    """
    Baseline which aggregates per-state baselines.
    """

    def __init__(self, name, baselines, inputs_spec, l2_regularization=None, summary_labels=None):
        """
        Aggregated baseline.

        Args:
            baselines: Dict of per-state baseline specification dicts
        """

        super().__init__(
            name=name, inputs_spec=inputs_spec, l2_regularization=l2_regularization,
            summary_labels=summary_labels
        )

        self.baselines = OrderedDict()
        from tensorforce.core.baselines import baseline_modules
        for name, baseline in baselines.items():  # turn to ordereddict in agent
            self.baselines[name] = self.add_module(
                name=(name + '-baseline'), module=baseline, modules=baseline_modules,
                inputs_spec=self.inputs_spec[name]
            )

        self.prediction = self.add_module(
            name='prediction', module='linear', modules=layer_modules, size=0,
            input_spec=dict(type='float', shape=(len(self.baselines),), batched=True)
        )

    def tf_predict(self, states, internals):
        predictions = list()
        for name, baseline in self.baselines.items():
            prediction = baseline.predict(states=states[name], internals=internals)
            predictions.append(prediction)
        return self.prediction.apply(x=tf.stack(values=predictions, axis=1))
