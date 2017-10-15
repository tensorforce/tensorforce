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
from tensorforce.models.model import Model
from tensorforce.models.distribution_model import DistributionModel
from tensorforce.models.pg_model import PGModel
from tensorforce.models.pg_log_prob_model import PGLogProbModel
from tensorforce.models.pg_prob_ratio_model import PGProbRatioModel
from tensorforce.models.q_model import QModel
from tensorforce.models.q_nstep_model import QNstepModel
from tensorforce.models.q_naf_model import QNAFModel
from tensorforce.models.q_demo_model import QDemoModel


models = dict(
    pg_log_prob_model=PGLogProbModel,
    pg_prob_ratio_model=PGProbRatioModel,
    q_model=QModel,
    q_nstep_model=QNstepModel,
    q_naf_model=QNAFModel,
    q_demo_model=QDemoModel
)


__all__ = [
    'Model',
    'DistributionModel',
    'PGModel',
    'PGProbRatioModel',
    'PGLogProbModel',
    'QModel',
    'QNstepModel',
    'QNAFModel',
    'QDemoModel',
    'models'
]
