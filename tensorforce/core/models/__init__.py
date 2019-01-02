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

from tensorforce.core.models.model import Model
from tensorforce.core.models.memory_model import MemoryModel
from tensorforce.core.models.distribution_model import DistributionModel
from tensorforce.core.models.pg_model import PGModel
from tensorforce.core.models.q_model import QModel


from tensorforce.core.models.constant_model import ConstantModel
from tensorforce.core.models.dpg_target_model import DPGTargetModel
from tensorforce.core.models.pg_log_prob_model import PGLogProbModel
from tensorforce.core.models.pg_prob_ratio_model import PGProbRatioModel
from tensorforce.core.models.q_demo_model import QDemoModel
from tensorforce.core.models.q_naf_model import QNAFModel
from tensorforce.core.models.q_nstep_model import QNstepModel
from tensorforce.core.models.random_model import RandomModel


models = dict(
    constant=ConstantModel, pg_log_prob=PGLogProbModel, pg_log_prob_target=DPGTargetModel,
    pg_prob_ratio=PGProbRatioModel, q_demo=QDemoModel, q=QModel, q_naf=QNAFModel,
    q_nstep=QNstepModel, random=RandomModel
)


__all__ = [
    'ConstantModel', 'DistributionModel', 'DPGTargetModel', 'MemoryModel', 'Model', 'models',
    'RandomModel', 'PGModel', 'PGLogProbModel', 'PGProbRatioModel', 'QDemoModel', 'QModel',
    'QNAFModel', 'QNstepModel', 'RandomModel'
]
