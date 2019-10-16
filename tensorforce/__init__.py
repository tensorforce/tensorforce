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

from tensorforce.exception import TensorforceError
from tensorforce import util


__all__ = ['TensorforceError', 'util']


__version__ = '0.5.2'

"""
pip install --upgrade pip setuptools wheel twine
python setup.py sdist bdist_wheel
twine upload --repository-url https://test.pypi.org/legacy/ dist/Tensorforce-0.5.2*
test: pip install --upgrade --index-url https://test.pypi.org/simple/ tensorforce
test: python; import tensorforce;
twine upload dist/Tensorforce-0.5.2*
"""


# Libraries should add NullHandler() by default, as its the application code's
# responsibility to configure log handlers.
# https://docs.python.org/3/howto/logging.html#configuring-logging-for-a-library

import logging

try:
    NullHandler = logging.NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

logging.getLogger(__name__).addHandler(NullHandler())
