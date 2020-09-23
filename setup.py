# Copyright 2020 Tensorforce Team. All Rights Reserved.
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


"""
cd docs; make html; cd ..;

pip install --upgrade -r requirements-all.txt
  ... update requirements.txt and setup.py ...

rm -r build
rm -r dist
rm -r docs/_*
pip install --upgrade pip setuptools wheel twine
python setup.py sdist bdist_wheel

twine upload --repository-url https://test.pypi.org/legacy/ dist/Tensorforce-0.6.*

cd ..
pip install --upgrade --index-url https://test.pypi.org/simple/ tensorforce
python
  > import tensorforce
  > print(tensorforce.__version__)
python tensorforce/examples/quickstart.py

cd tensorforce
twine upload dist/Tensorforce-0.6.*
  ... commit and fix GitHub version ...
"""

if sys.version_info.major != 3:
    raise NotImplementedError("Tensorforce is only compatible with Python 3.")

tensorforce_directory = os.path.abspath(os.path.dirname(__file__))

# Extract version from tensorforce/__init__.py
version = None
with open(os.path.join(tensorforce_directory, 'tensorforce', '__init__.py'), 'r') as filehandle:
    for line in filehandle:
        if line.startswith('__version__ = \'') and line.endswith('\'\n'):
            version = line[15:-2]
assert version is not None

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
assert len(long_description) > 0
long_description.append('\n')
long_description.append('For more information, see the [GitHub project page](https://github.com/ten'
                        'sorforce/tensorforce) and [ReadTheDocs documentation](https://tensorforce.'
                        'readthedocs.io/en/latest/).\n')
long_description = ''.join(long_description)

# Find packages
packages = find_packages(exclude=('test',))
assert all(package.startswith('tensorforce') for package in packages)

# Extract install_requires from requirements.txt
install_requires = list()
with open(os.path.join(tensorforce_directory, 'requirements.txt'), 'r') as filehandle:
    for line in filehandle:
        line = line.strip()
        if line:
            install_requires.append(line)
assert len(install_requires) > 0

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
    packages=packages,
    download_url='https://github.com/tensorforce/tensorforce/archive/{}.tar.gz'.format(version),
    license='Apache 2.0',
    python_requires='>=3.5',
    classifiers=[
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    install_requires=install_requires,
    extras_require=dict(
        tfa=['tensorflow-addons >= 0.11.2'],
        tune=['hpbandster >= 0.7.4'],
        envs=[
            'ale-py', 'gym[atari,box2d,classic_control] >= 0.17.2', 'box2d >= 2.3.10',
            'gym-retro >= 0.8.0', 'vizdoom >= 1.1.7'
        ],
        ale=['ale-py'],
        gym=['gym[atari,box2d,classic_control] >= 0.17.2', 'box2d >= 2.3.10'],
        retro=['gym-retro >= 0.8.0'],
        vizdoom=['vizdoom >= 1.1.7'],
        carla=['pygame', 'opencv-python'],
        docs=[
            'm2r >= 0.2.1', 'recommonmark >= 0.6.0', 'sphinx >= 3.2.1', 'sphinx-rtd-theme >= 0.5.0'
        ]
    ),
    zip_safe=False
)
