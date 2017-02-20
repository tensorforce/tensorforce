# Copyright 2016 reinforce.io. All Rights Reserved.
# ==============================================================================

"""
TensorForce exceptions
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


class TensorForceError(Exception):
    pass


class TensorForceValueError(TensorForceError):
    pass


class ArgumentMustBePositiveError(TensorForceValueError):
    pass


class ConfigError(TensorForceValueError):
    pass
