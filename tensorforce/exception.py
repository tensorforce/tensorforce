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


def is_iterable(x):
    if isinstance(x, str):
        return False
    try:
        iter(x)
        return True
    except TypeError:
        return False


class TensorforceError(Exception):
    """
    Tensorforce error
    """

    def __init__(self, message):
        if message[0].islower():
            message = message[0].upper() + message[1:]
        if message[-1] != '.':
            message = message + '.'
        super().__init__(message)

    @staticmethod
    def unexpected():
        return TensorforceError(message="Unexpected error!")

    @staticmethod
    def collision(name, value, group1, group2):
        return TensorforceError(
            message="{name} collision between {group1} and {group2}: {value}.".format(
                name=name, group1=group1, group2=group2, value=value
            )
        )

    @staticmethod
    def mismatch(name, value1, value2, argument=None):
        if argument is None:
            return TensorforceError(
                message="{name} mismatch: {value1} <-> {value2}.".format(
                    name=name, value1=value1, value2=value2
                )
            )
        else:
            return TensorforceError(
                message="{name} mismatch for argument {argument}: {value1} <-> {value2}.".format(
                    name=name, argument=argument, value1=value1, value2=value2
                )
            )

    @staticmethod
    def exists(name, value):
        return TensorforceError(
            message="{name} already exists: {value}.".format(name=name, value=value)
        )

    @staticmethod
    def required(name, value):
        if is_iterable(x=value):
            value = ', '.join(str(x) for x in value)
        return TensorforceError(
            message="Missing {name} value: {value}.".format(name=name, value=value)
        )

    @staticmethod
    def type(name, value, argument=None):
        if argument is None:
            return TensorforceError(
                message="Invalid {name} type: {type}.".format(name=name, type=type(value))
            )
        else:
            return TensorforceError(
                message="Invalid type for {name} argument {argument}: {type}.".format(
                    name=name, argument=argument, type=type(value)
                )
            )

    @staticmethod
    def value(name, value, argument=None):
        if is_iterable(x=value):
            value = ', '.join(str(x) for x in value)
        if argument is None:
            return TensorforceError(
                message="Invalid {name} value: {value}.".format(name=name, value=value)
            )
        else:
            return TensorforceError(
                message="Invalid value for {name} argument {argument}: {value}.".format(
                    name=name, argument=argument, value=value
                )
            )
