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
        if message[-1] not in '.!?':
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
                message="{name} mismatch: {value2} != {value1}.".format(
                    name=name, value1=value1, value2=value2
                )
            )
        else:
            return TensorforceError(
                message="{name} mismatch for argument {argument}: {value2} != {value1}.".format(
                    name=name, argument=argument, value1=value1, value2=value2
                )
            )

    @staticmethod
    def exists(name, value):
        return TensorforceError(
            message="{name} already exists: {value}.".format(name=name, value=value)
        )

    @staticmethod
    def exists_not(name, value):
        return TensorforceError(
            message="{name} does not exist: {value}.".format(name=name, value=value)
        )

    @staticmethod
    def required_attribute(name, attribute):
        return TensorforceError(
            message="Required {name} attribute {attribute}.".format(name=name, attribute=attribute)
        )

    @staticmethod
    def required(name, argument, expected=None, condition=None):
        if condition is None:
            if expected is None:
                return TensorforceError(
                    message="Required {name} argument {argument}.".format(
                        name=name, argument=argument
                    )
                )
            else:
                return TensorforceError(
                    message="Required {name} argument {argument} to be {expected}.".format(
                        name=name, argument=argument, expected=expected
                    )
                )
        else:
            if expected is None:
                return TensorforceError(
                    message="Required {name} argument {argument} given {condition}.".format(
                        name=name, argument=argument, condition=condition
                    )
                )
            else:
                return TensorforceError(
                    message="Required {name} argument {argument} to be {expected} given "
                            "{condition}.".format(
                        name=name, argument=argument, expected=expected, condition=condition
                    )
                )

    @staticmethod
    def invalid(name, argument, condition=None):
        if condition is None:
            return TensorforceError(
                message="Invalid {name} argument {argument}.".format(name=name, argument=argument)
            )
        else:
            return TensorforceError(
                message="Invalid {name} argument {argument} given {condition}.".format(
                    name=name, condition=condition, argument=argument
                )
            )

    @staticmethod
    def type(name, argument, dtype, condition=None, hint=None):
        if hint is None:
            if condition is None:
                return TensorforceError(
                    message="Invalid type for {name} argument {argument}: {type}.".format(
                        name=name, argument=argument, type=dtype
                    )
                )
            else:
                return TensorforceError(
                    message="Invalid type for {name} argument {argument} given {condition}: {type}.".format(
                        name=name, argument=argument, condition=condition, type=dtype
                    )
                )
        else:
            if condition is None:
                return TensorforceError(
                    message="Invalid type for {name} argument {argument}: {type} {hint}.".format(
                        name=name, argument=argument, type=dtype, hint=hint
                    )
                )
            else:
                return TensorforceError(
                    message="Invalid type for {name} argument {argument} given {condition}: {type} {hint}.".format(
                        name=name, argument=argument, condition=condition, type=dtype, hint=hint
                    )
                )

    @staticmethod
    def value(name, argument, value, condition=None, hint=None):
        if isinstance(value, dict):
            value = str(value)
        elif is_iterable(x=value):
            value = ','.join(str(x) for x in value)
        if hint is None:
            if condition is None:
                return TensorforceError(
                    message="Invalid value for {name} argument {argument}: {value}.".format(
                        name=name, argument=argument, value=value
                    )
                )
            else:
                return TensorforceError(
                    message="Invalid value for {name} argument {argument} given {condition}: {value}.".format(
                        name=name, argument=argument, condition=condition, value=value
                    )
                )
        else:
            if condition is None:
                return TensorforceError(
                    message="Invalid value for {name} argument {argument}: {value} {hint}.".format(
                        name=name, argument=argument, value=value, hint=hint
                    )
                )
            else:
                return TensorforceError(
                    message="Invalid value for {name} argument {argument} given {condition}: {value} {hint}.".format(
                        name=name, argument=argument, condition=condition, value=value, hint=hint
                    )
                )

    @staticmethod
    def deprecated(name, argument, replacement):
        return DeprecationWarning(
            "Deprecated {name} argument {argument}, use {replacement} instead.".format(
                name=name, argument=argument, replacement=replacement
            )
        )
