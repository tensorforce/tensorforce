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

import tensorflow as tf

from tensorforce import util
from tensorforce.core import Module
from tensorforce.core.parameters import Parameter


class Decaying(Parameter):
    """
    Decaying hyperparameter.

    Args:
        name (string): Module name
            (<span style="color:#0000C0"><b>internal use</b></span>).
        dtype ("bool" | "int" | "long" | "float"): Tensor type
            (<span style="color:#0000C0"><b>internal use</b></span>).
        unit ("timesteps" | "episodes" | "updates"): Unit of decay schedule
            (<span style="color:#C00000"><b>required</b></span>).
        decay ("cosine" | "cosine_restarts" | "exponential" | "inverse_time" | "linear_cosine" | "linear_cosine_noisy" | "polynomial"):
            Decay type, see
            `TensorFlow docs <https://www.tensorflow.org/api_docs/python/tf/train>`__
            (<span style="color:#C00000"><b>required</b></span>).
        initial_value (float): Initial value
            (<span style="color:#C00000"><b>required</b></span>).
        decay_steps (long): Number of decay steps
            (<span style="color:#C00000"><b>required</b></span>).
        increasing (bool): Whether to subtract the decayed value from 1.0
            (<span style="color:#00C000"><b>default</b></span>: false).
        inverse (bool): Whether to take the inverse of the decayed value
            (<span style="color:#00C000"><b>default</b></span>: false).
        scale (float): Scaling factor for (inverse) decayed value
            (<span style="color:#00C000"><b>default</b></span>: 1.0).
        min_value (dtype-compatible value): Lower parameter value bound
            (<span style="color:#0000C0"><b>internal use</b></span>).
        max_value (dtype-compatible value): Upper parameter value bound
            (<span style="color:#0000C0"><b>internal use</b></span>).
        summary_labels ("all" | iter[string]): Labels of summaries to record
            (<span style="color:#00C000"><b>default</b></span>: inherit value of parent module).
        kwargs: Additional arguments depend on decay mechanism.<br>
            Cosine decay:
            <ul>
            <li><b>alpha</b> (<i>float</i>) &ndash; Minimum learning rate value as a fraction of
            learning_rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).</li>
            </ul>
            Cosine decay with restarts:
            <ul>
            <li><b>t_mul</b> (<i>float</i>) &ndash; Used to derive the number of iterations in the
            i-th period
            (<span style="color:#00C000"><b>default</b></span>: 2.0).</li>
            <li><b>m_mul</b> (<i>float</i>) &ndash; Used to derive the initial learning rate of the
            i-th period
            (<span style="color:#00C000"><b>default</b></span>: 1.0).</li>
            <li><b>alpha</b> (<i>float</i>) &ndash; Minimum learning rate value as a fraction of
            the learning_rate
            (<span style="color:#00C000"><b>default</b></span>: 0.0).</li>
            </ul>
            Exponential decay:
            <ul>
            <li><b>decay_rate</b> (<i>float</i>) &ndash; Decay rate
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>staircase</b> (<i>bool</i>) &ndash; Whether to apply decay in a discrete
            staircase, as opposed to continuous, fashion.
            (<span style="color:#00C000"><b>default</b></span>: false).</li>
            </ul>
            Inverse time decay:
            <ul>
            <li><b>decay_rate</b> (<i>float</i>) &ndash; Decay rate
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>staircase</b> (<i>bool</i>) &ndash; Whether to apply decay in a discrete
            staircase, as opposed to continuous, fashion.
            (<span style="color:#00C000"><b>default</b></span>: false).</li>
            </ul>
            Linear cosine decay:
            <ul>
            <li><b>num_periods</b> (<i>float</i>) &ndash; Number of periods in the cosine part of
            the decay
            (<span style="color:#00C000"><b>default</b></span>: 0.5).</li>
            <li><b>alpha</b> (<i>float</i>) &ndash; Alpha value
            (<span style="color:#00C000"><b>default</b></span>: 0.0).</li>
            <li><b>beta</b> (<i>float</i>) &ndash; Beta value
            (<span style="color:#00C000"><b>default</b></span>: 0.001).</li>
            </ul>
            Natural exponential decay:
            <ul>
            <li><b>decay_rate</b> (<i>float</i>) &ndash; Decay rate
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>staircase</b> (<i>bool</i>) &ndash; Whether to apply decay in a discrete
            staircase, as opposed to continuous, fashion.
            (<span style="color:#00C000"><b>default</b></span>: false).</li>
            </ul>
            Noisy linear cosine decay:
            <ul>
            <li><b>initial_variance</b> (<i>float</i>) &ndash; Initial variance for the noise
            (<span style="color:#00C000"><b>default</b></span>: 1.0).</li>
            <li><b>variance_decay</b> (<i>float</i>) &ndash; Decay for the noise's variance
            (<span style="color:#00C000"><b>default</b></span>: 0.55).</li>
            <li><b>num_periods</b> (<i>float</i>) &ndash; Number of periods in the cosine part of
            the decay
            (<span style="color:#00C000"><b>default</b></span>: 0.5).</li>
            <li><b>alpha</b> (<i>float</i>) &ndash; Alpha value
            (<span style="color:#00C000"><b>default</b></span>: 0.0).</li>
            <li><b>beta</b> (<i>float</i>) &ndash; Beta value
            (<span style="color:#00C000"><b>default</b></span>: 0.001).</li>
            </ul>
            Polynomial decay:
            <ul>
            <li><b>final_value</b> (<i>float</i>) &ndash; Final value
            (<span style="color:#C00000"><b>required</b></span>).</li>
            <li><b>power</b> (<i>float</i>) &ndash; Power of polynomial
            (<span style="color:#00C000"><b>default</b></span>: 1.0, thus linear).</li>
            <li><b>cycle</b> (<i>bool</i>) &ndash; Whether to cycle beyond decay_steps
            (<span style="color:#00C000"><b>default</b></span>: false).</li>
            </ul>
    """

    def __init__(
        self, name, dtype, unit, decay, initial_value, decay_steps, increasing=False,
        inverse=False, scale=1.0, min_value=None, max_value=None, summary_labels=None, **kwargs
    ):
        assert unit in ('timesteps', 'episodes', 'updates')
        assert decay in (
            'cosine', 'cosine_restarts', 'exponential', 'inverse_time', 'linear_cosine',
            'linear_cosine_noisy', 'polynomial'
        )
        assert isinstance(initial_value, float)
        assert isinstance(decay_steps, int) or decay_steps % 10.0 == 0.0

        self.decay = decay
        self.initial_value = initial_value
        self.decay_steps = int(decay_steps)
        self.increasing = increasing
        self.inverse = inverse
        self.scale = scale
        self.kwargs = kwargs

        super().__init__(
            name=name, dtype=dtype, unit=unit, min_value=min_value, max_value=max_value,
            summary_labels=summary_labels
        )

    def min_value(self):
        if self.decay == 'cosine' or self.decay == 'cosine_restarts':
            assert 0.0 <= self.kwargs.get('alpha', 0.0) <= 1.0
            if self.initial_value >= 0.0:
                min_value = self.initial_value * self.kwargs.get('alpha', 0.0)
                max_value = self.initial_value
            else:
                min_value = self.initial_value
                max_value = self.initial_value * self.kwargs.get('alpha', 0.0)

        elif self.decay == 'exponential' or self.decay == 'inverse_time':
            assert 0.0 <= self.kwargs['decay_rate'] <= 1.0
            if self.kwargs['decay_rate'] == 1.0:
                min_value = max_value = self.initial_value
            elif self.initial_value >= 0.0:
                min_value = 0.0
                max_value = self.initial_value
            else:
                min_value = self.initial_value
                max_value = 0.0

        elif self.decay == 'linear_cosine' or self.decay == 'linear_cosine_noisy':
            assert 0.0 <= self.kwargs.get('alpha', 0.0) <= 1.0
            assert 0.0 <= self.kwargs.get('beta', 0.0) <= 1.0
            if self.initial_value >= 0.0:
                min_value = self.initial_value * self.kwargs.get('beta', 0.001)
                max_value = self.initial_value * (
                    1.0 + self.kwargs.get('alpha', 0.0) + self.kwargs.get('beta', 0.001)
                )
            else:
                min_value = self.initial_value * (
                    1.0 + self.kwargs.get('alpha', 0.0) + self.kwargs.get('beta', 0.001)
                )
                max_value = self.initial_value * self.kwargs.get('beta', 0.001)

        elif self.decay == 'polynomial':
            if self.kwargs.get('power', 1.0) == 0.0:
                min_value = max_value = self.initial_value
            elif self.initial_value >= self.kwargs['final_value']:
                min_value = self.kwargs['final_value']
                max_value = self.initial_value
            else:
                min_value = self.initial_value
                max_value = self.kwargs['final_value']

        assert min_value <= max_value

        if self.increasing:
            assert 0.0 <= min_value <= max_value <= 1.0
            min_value, max_value = 1.0 - max_value, 1.0 - min_value

        if self.inverse:
            assert 0.0 < min_value <= max_value
            min_value, max_value = 1.0 / max_value, 1.0 / min_value

        if self.scale == 1.0:
            pass
        elif self.scale >= 0.0:
            min_value, max_value = self.scale * min_value, self.scale * max_value
        else:
            min_value, max_value = self.scale * max_value, self.scale * min_value

        return util.py_dtype(dtype=self.dtype)(min_value)

    def max_value(self):
        if self.decay == 'cosine' or self.decay == 'cosine_restarts':
            assert 0.0 <= self.kwargs.get('alpha', 0.0) <= 1.0
            if self.initial_value >= 0.0:
                min_value = self.initial_value * self.kwargs.get('alpha', 0.0)
                max_value = self.initial_value
            else:
                min_value = self.initial_value
                max_value = self.initial_value * self.kwargs.get('alpha', 0.0)

        elif self.decay == 'exponential' or self.decay == 'inverse_time':
            assert 0.0 <= self.kwargs['decay_rate'] <= 1.0
            if self.kwargs['decay_rate'] == 1.0:
                min_value = max_value = self.initial_value
            elif self.initial_value >= 0.0:
                min_value = 0.0
                max_value = self.initial_value
            else:
                min_value = self.initial_value
                max_value = 0.0

        elif self.decay == 'linear_cosine' or self.decay == 'linear_cosine_noisy':
            assert 0.0 <= self.kwargs.get('alpha', 0.0) <= 1.0
            assert 0.0 <= self.kwargs.get('beta', 0.0) <= 1.0
            if self.initial_value >= 0.0:
                min_value = self.initial_value * self.kwargs.get('beta', 0.001)
                max_value = self.initial_value * (
                    1.0 + self.kwargs.get('alpha', 0.0) + self.kwargs.get('beta', 0.001)
                )
            else:
                min_value = self.initial_value * (
                    1.0 + self.kwargs.get('alpha', 0.0) + self.kwargs.get('beta', 0.001)
                )
                max_value = self.initial_value * self.kwargs.get('beta', 0.001)

        elif self.decay == 'polynomial':
            if self.kwargs.get('power', 1.0) == 0.0:
                min_value = max_value = self.initial_value
            elif self.initial_value >= self.kwargs['final_value']:
                min_value = self.kwargs['final_value']
                max_value = self.initial_value
            else:
                min_value = self.initial_value
                max_value = self.kwargs['final_value']

        assert min_value <= max_value

        if self.increasing:
            assert 0.0 <= min_value <= max_value <= 1.0
            min_value, max_value = 1.0 - max_value, 1.0 - min_value

        if self.inverse:
            assert 0.0 < min_value <= max_value
            min_value, max_value = 1.0 / max_value, 1.0 / min_value

        if self.scale == 1.0:
            pass
        elif self.scale >= 0.0:
            min_value, max_value = self.scale * min_value, self.scale * max_value
        else:
            min_value, max_value = self.scale * max_value, self.scale * min_value

        return util.py_dtype(dtype=self.dtype)(max_value)

    def final_value(self):
        if self.decay == 'cosine' or self.decay == 'cosine_restarts':
            assert 0.0 <= self.kwargs['decay_rate'] <= 1.0
            value = self.initial_value * self.kwargs.get('alpha', 0.0)

        elif self.decay == 'exponential' or self.decay == 'inverse_time':
            assert 0.0 <= self.kwargs['decay_rate'] <= 1.0
            if self.kwargs['decay_rate'] == 1.0:
                value = self.initial_value
            else:
                value = 0.0

        elif self.decay == 'linear_cosine' or self.decay == 'linear_cosine_noisy':
            assert 0.0 <= self.kwargs.get('alpha', 0.0) <= 1.0
            assert 0.0 <= self.kwargs.get('beta', 0.0) <= 1.0
            value = self.initial_value * self.kwargs.get('beta', 0.001)

        elif self.decay == 'polynomial':
            if self.kwargs.get('power', 1.0) == 0.0:
                value = self.initial_value
            else:
                value = self.kwargs['final_value']

        if self.increasing:
            assert 0.0 <= value <= 1.0
            value = 1.0 - value

        if self.inverse:
            assert value > 0.0
            value = 1.0 / value

        if self.scale != 1.0:
            value = value * self.scale

        return util.py_dtype(dtype=self.dtype)(value)


    def parameter_value(self, step):
        initial_value = tf.constant(value=self.initial_value, dtype=util.tf_dtype(dtype='float'))

        if self.decay == 'cosine':
            assert 0.0 <= self.kwargs.get('alpha', 0.0) <= 1.0
            parameter = tf.keras.experimental.CosineDecay(
                initial_learning_rate=initial_value, decay_steps=self.decay_steps,
                alpha=self.kwargs.get('alpha', 0.0)
            )(step=step)

        elif self.decay == 'cosine_restarts':
            assert 0.0 <= self.kwargs.get('alpha', 0.0) <= 1.0
            parameter = tf.keras.experimental.CosineDecayRestarts(
                initial_learning_rate=initial_value, first_decay_steps=self.decay_steps,
                t_mul=self.kwargs.get('t_mul', 2.0), m_mul=self.kwargs.get('m_mul', 1.0),
                alpha=self.kwargs.get('alpha', 0.0)
            )(step=step)

        elif self.decay == 'exponential':
            assert self.kwargs['decay_rate'] >= 0.0
            parameter = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=initial_value, decay_steps=self.decay_steps,
                decay_rate=self.kwargs['decay_rate'], staircase=self.kwargs.get('staircase', False)
            )(step=step)

        elif self.decay == 'inverse_time':
            assert self.kwargs['decay_rate'] >= 0.0
            parameter = tf.keras.optimizers.schedules.InverseTimeDecay(
                initial_learning_rate=initial_value, decay_steps=self.decay_steps,
                decay_rate=self.kwargs['decay_rate'], staircase=self.kwargs.get('staircase', False)
            )(step=step)

        elif self.decay == 'linear_cosine':
            assert self.kwargs.get('beta', 0.001) >= 0.0
            parameter = tf.keras.experimental.LinearCosineDecay(
                initial_learning_rate=initial_value, decay_steps=self.decay_steps,
                num_periods=self.kwargs.get('num_periods', 0.5),
                alpha=self.kwargs.get('alpha', 0.0), beta=self.kwargs.get('beta', 0.001)
            )(step=step)

        elif self.decay == 'linear_cosine_noisy':
            assert self.kwargs.get('beta', 0.001) >= 0.0
            parameter = tf.keras.experimental.NoisyLinearCosineDecay(
                initial_learning_rate=initial_value, decay_steps=self.decay_steps,
                initial_variance=self.kwargs.get('initial_variance', 1.0),
                variance_decay=self.kwargs.get('variance_decay', 0.55),
                num_periods=self.kwargs.get('num_periods', 0.5),
                alpha=self.kwargs.get('alpha', 0.0), beta=self.kwargs.get('beta', 0.001)
            )(step=step)

        elif self.decay == 'polynomial':
            assert self.kwargs['power'] >= 0.0
            parameter = tf.keras.optimizers.schedules.PolynomialDecay(
                initial_learning_rate=initial_value, decay_steps=self.decay_steps,
                end_learning_rate=self.kwargs['final_value'], power=self.kwargs.get('power', 1.0),
                cycle=self.kwargs.get('cycle', False)
            )(step=step)

        if self.increasing:
            one = tf.constant(value=1.0, dtype=util.tf_dtype(dtype='float'))
            parameter = one - parameter

        if self.inverse:
            one = tf.constant(value=1.0, dtype=util.tf_dtype(dtype='float'))
            parameter = one / parameter

        if self.scale != 1.0:
            scale = tf.constant(value=self.scale, dtype=util.tf_dtype(dtype='float'))
            parameter = parameter * scale

        if self.dtype != 'float':
            parameter = tf.dtypes.cast(x=parameter, dtype=util.tf_dtype(dtype=self.dtype))

        return parameter
