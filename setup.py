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
from setuptools import find_packages, setup
import sys


if sys.version_info.major != 3:
    raise NotImplementedError("Tensorforce is only compatible with Python 3.")


tensorforce_directory = os.path.abspath(os.path.dirname(__file__))

# Extract version from tensorforce/__init__.py
with open(os.path.join(tensorforce_directory, 'tensorforce', '__init__.py'), 'r') as filehandle:
    for line in filehandle:
        if line.startswith('__version__'):
            version = line[15:-2]

# Extract long_description from README.md introduction
long_description = list()
with open(os.path.join(tensorforce_directory, 'README.md'), 'r') as filehandle:
    lines = iter(filehandle)
    line = next(lines)
    if not line.startswith('# Tensorforce:'):
        raise NotImplementedError
    long_description.append(line)
    for line in lines:
        if line == '#### Introduction\n':
            break
    if next(lines) != '\n':
        raise NotImplementedError
    while True:
        line = next(lines)
        if line == '\n':
            line = next(lines)
            if line == '\n':
                break
            else:
                long_description.append('\n')
                long_description.append(line)
        else:
            long_description.append(line)
    while line == '\n':
        line = next(lines)
    if not line.startswith('#### '):
        raise NotImplementedError
long_description = ''.join(long_description)


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
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Alexander Kuhnle',
    author_email='tensorforce.team@gmail.com',
    url='http://github.com/tensorforce/tensorforce',
    packages=[
        package for package in find_packages(exclude=('test',))
        if package.startswith('tensorforce')
    ],
    download_url='https://github.com/tensorforce/tensorforce/archive/{}.tar.gz'.format(version),
    license='Apache 2.0',
    python_requires='>=3.5',
    install_requires=install_requires,
    extras_require=dict(
        tf=['tensorflow'],
        tf_gpu=['tensorflow-gpu'],
        tfa=['tensorflow-addons'],
        docs=['m2r', 'recommonmark', 'sphinx', 'sphinx-rtd-theme'],
        envs=['gym[all]', 'gym-retro', 'mazeexp', 'vizdoom'],
        mazeexp=['mazeexp'],
        gym=['gym[all]'],
        retro=['gym-retro'],
        vizdoom=['vizdoom']
    ),
    zip_safe=False
)
