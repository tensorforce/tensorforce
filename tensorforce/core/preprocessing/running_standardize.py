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
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorforce import util
from tensorforce.core.preprocessing import Preprocessor


class RunningStandardize(Preprocessor):
    """
    Standardize state w.r.t past states. Subtract mean and divide by standard deviation of sequence of past states.
    """

    def __init__(self, axis=None, reset_after_batch=True, scope='running_standardize', summary_labels=()):
        self.axis = axis
        self.reset_after_batch = reset_after_batch
        self.history = list()
        super(RunningStandardize, self).__init__(scope=scope, summary_labels=summary_labels)

    def reset(self):
        if self.reset_after_batch:
            self.history = list()

    def tf_process(self, tensor):
        state = tensor.astype(np.float32)
        self.history.append(state)
        history = np.array(self.history)

        return (state - history.mean(axis=self.axis)) / (state.std(axis=self.axis) + util.epsilon)
