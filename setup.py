# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup

setup(name='tensorforce',
      version='0.1',
      description='Reinforcement learning for TensorFlow',
      url='http://github.com/reinforceio/tensorforce',
      author='reinforce.io',
      author_email='contact@reinforce.io',
      license='Apache 2.0',
      packages=['tensorforce'],
      install_requires=[
          'tensorflow',
          'numpy',
          'six',
          'scipy',
          'nose'
      ],
      zip_safe=False)
