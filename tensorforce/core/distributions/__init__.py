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


from tensorforce.core.distributions.distribution import Distribution
from tensorforce.core.distributions.bernoulli import Bernoulli
from tensorforce.core.distributions.categorical import Categorical
from tensorforce.core.distributions.gaussian import Gaussian
from tensorforce.core.distributions.beta import Beta


distributions = dict(
    bernoulli=Bernoulli,
    categorical=Categorical,
    gaussian=Gaussian,
    beta=Beta
)


__all__ = ['distributions', 'Distribution', 'Bernoulli', 'Categorical', 'Gaussian', 'Beta']
