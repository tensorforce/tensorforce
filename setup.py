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

import os
from setuptools import setup


tensorforce_directory = os.path.abspath(os.path.dirname(__file__))

# Extract version from tensorforce/__init__.py
with open(os.path.join(tensorforce_directory, 'tensorforce', '__init__.py'), 'r') as filehandle:
    for line in filehandle:
        if line.startswith('__version__'):
            version = line[15:-2]

# Extract install_requires from requirements.txt
install_requires = list()
with open(os.path.join(tensorforce_directory, 'requirements.txt'), 'r') as filehandle:
    for line in filehandle:
        line = line.strip()
        if line:
            install_requires.append(line)

# Readthedocs requires Sphinx extensions to be specified as part of install_requires.
if os.environ.get('READTHEDOCS', None) == 'True':
    install_requires.append('recommonmark')

setup(
    name='Tensorforce',
    version=version,
    description='Tensorforce: a TensorFlow library for applied reinforcement learning',
    author='Tensorforce Team',
    author_email='tensorforce.team@gmail.com',
    url='http://github.com/reinforceio/tensorforce',
    packages=['tensorforce', 'examples'],
    download_url='https://github.com/reinforceio/tensorforce/archive/0.5.0.tar.gz',
    license='Apache 2.0',
    install_requires=install_requires,
    setup_requires=['numpy', 'recommonmark'],  # ???????????
    extras_require=dict(
        tf=["tensorflow>=1.6.0"],
        tf_gpu=["tensorflow-gpu>=1.6.0"]
    ),
    zip_safe=False  # ???????????
)
