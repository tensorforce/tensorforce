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

from tensorforce import util
import tensorforce.core.explorations


class Exploration(object):

    def __call__(self, episode=0, timestep=0):
        raise NotImplementedError

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
