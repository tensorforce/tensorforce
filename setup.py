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

import os

from setuptools import setup

install_requires=[
          'tensorflow',
          'numpy',
          'six',
          'scipy',
          'pillow',
          'pytest'
      ]

setup_requires=['numpy', 'mistune']

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    install_requires.extend(setup_requires)


setup(name='tensorforce',
      version='0.2',
      description='Reinforcement learning for TensorFlow',
      url='http://github.com/reinforceio/tensorforce',
      author='reinforce.io',
      author_email='contact@reinforce.io',
      license='Apache 2.0',
      packages=['tensorforce'],
      install_requires=install_requires,
      setup_requires=setup_requires,
      zip_safe=False)
