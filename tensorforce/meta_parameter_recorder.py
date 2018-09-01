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

import inspect
import os
import numpy as np
import tensorflow as tf
from tensorforce import TensorForceError


class MetaParameterRecorder(object):
    """
    Class to record MetaParameters as well as Summary/Description for TensorBoard (TEXT & FILE will come later).

    General:

    * format_type: used to configure data conversion for TensorBoard=0, TEXT & JSON (not Implemented), etc
    """

    def __init__(self, current_frame):
        """
        Init the MetaPrameterRecord with "Agent" parameters by passing inspect.currentframe() from Agent Class.

        The Init will search back to find the parent class to capture all passed parameters and store
        them in "self.meta_params".

        NOTE: Currently only optimized for TensorBoard output.

        TODO: Add JSON Export, TEXT EXPORT

        Args:
            current_frame: Frame value from class to obtain metaparameters[= inspect.currentframe()]

        """
        self.ignore_unknown_dtypes = False
        self.meta_params = dict()
        self.method_calling = inspect.getframeinfo(current_frame)[2]

        _, _, __, self.vals_current = inspect.getargvalues(current_frame)
        # self is the class name of the frame involved
        if 'self' in self.vals_current:
            self.recorded_class_type = self.vals_current['self']
            # Add explicit AgentName item so class can be deleted
            self.meta_params['AgentName'] = str(self.vals_current['self'])

        frame_list = inspect.getouterframes(current_frame)

        for frame in frame_list:
            # Rather than frame.frame (named tuple), use [0] for python2.
            args, varargs, keywords, vals = inspect.getargvalues(frame[0])
            if 'self' in vals:
                if self.recorded_class_type == vals['self']:
                    for i in args:
                        self.meta_params[i] = vals[i]
        # Remove the "CLASS" from the dictionary, has no value "AgentName" contains STR of Class.
        del self.meta_params['self']

    def merge_custom(self, custom_dict):
        if type(custom_dict) is not dict:
            raise TensorForceError(
                "Error:  MetaParameterRecorder 'meta_dict' must be passed a dictionary "
                "but was passed a type {} which is not supported.".format(str(type(custom_dict)))
            )
        for key in custom_dict:
            if key in self.meta_params:
                raise TensorForceError(
                    "Error:  MetaParameterRecorder 'meta_dict' key {} conflicts with internal key,"
                    " please change passed key.".format(str(key))
                )
            self.meta_params[key] = custom_dict[key]
        # This line assumes the merge data came from summary_spec['meta_dict'], remove this from summary_spec.
        del self.meta_params['summarizer']['meta_dict']

    def text_output(self, format_type=1):
        print('======================= ' + self.meta_params['AgentName'] + ' ====================================')
        for key in self.meta_params:
            print(
                "    ",
                key,
                type(self.meta_params[key]),
                "=",
                self.convert_data_to_string(self.meta_params[key], format_type=format_type)
            )

        print('======================= ' + self.meta_params['AgentName'] + ' ====================================')

    def convert_dictionary_to_string(self, data, indent=0, format_type=0, separator=None, eol=None):
        data_string = ""
        add_separator = ""
        if eol is None:
            eol = os.linesep
        if separator is None:
            separator = ", "

        # This should not ever occur but here as a catch.
        if type(data) is not dict:
            raise TensorForceError(
                "Error:  MetaParameterRecorder Dictionary conversion was passed a type {}"
                " not supported.".format(str(type(data)))
            )

        # TensorBoard
        if format_type == 0:
            label = ""
            div = ""

            if indent > 0:
                label = "    | "
                div = "--- | "
            data_string += label + "Key | Value" + eol + div + "--- | ----" + eol

        for key in data:
            key_txt = key
            # TensorBoard
            if format_type == 0:
                key_txt = "**" + key + "**"
                key_value_sep = ' | '
                if indent > 0:
                    key_txt = "    | " + key_txt

            converted_data = self.convert_data_to_string(data[key], separator=separator, indent=indent+1)
            data_string += add_separator + key_txt + key_value_sep + converted_data + eol

        return data_string

    def convert_list_to_string(self, data, indent=0, format_type=0, eol=None, count=True):
        data_string = ""
        if eol is None:
            eol = os.linesep

        # This should not ever occur but here as a catch.
        if type(data) is not list:
            raise TensorForceError(
                "Error:  MetaParameterRecorder List conversion was passed a type {}"
                " not supported.".format(str(type(data)))
            )

        for index, line in enumerate(data):
            data_string_prefix = ""
            if count and indent == 0:
                data_string_prefix = str(index + 1) + ". "
            # TensorBoard
            if format_type == 0:
                # Only add indent for 2nd item and beyond as this is likely a dictionary entry.
                if indent > 0 and index > 0:
                    data_string_prefix = "    | " + data_string_prefix
            if index == (len(data) - 1):
                append_eol = ""
            else:
                append_eol = eol
            data_string += data_string_prefix + self.convert_data_to_string(line, indent=indent+1) + append_eol

        return data_string

    def convert_ndarray_to_md(self, data, format_type=0, eol=None):
        data_string = ""
        data_string1 = "|Row|"
        data_string2 = "|:---:|"
        if eol is None:
            eol = os.linesep

        # This should not ever occur but here as a catch.
        if type(data) is not np.ndarray:
            raise TensorForceError(
                "Error:  MetaParameterRecorder ndarray conversion was passed"
                " a type {} not supported.".format(str(type(data)))
            )

        shape = data.shape
        rank = data.ndim

        if rank == 2:
            for col in range(shape[1]):
                data_string1 += "Col-" + str(col) + "|"
                data_string2 += ":----:|"
            data_string += data_string1 + eol + data_string2 + eol

            for row in range(shape[0]):
                data_string += "|" + str(row) + "|"
                for col in range(shape[1]):
                    data_string += str(data[row, col]) + "|"

                if row != (shape[0] - 1):
                    data_string += eol

        elif rank == 1:
            data_string += "|Row|Col-0|" + eol + "|:----:|:----:|" + eol

            for row in range(shape[0]):
                data_string += str(row) + "|" + str(data[row]) + "|" + eol

        return data_string

    def convert_data_to_string(self, data, indent=0, format_type=0, separator=None, eol=None):
        data_string = ""
        if type(data) is int:
            data_string = str(data)
        elif type(data) is float:
            data_string = str(data)
        elif type(data) is str:
            data_string = data
        elif type(data) is tuple:
            data_string = str(data)
        elif type(data) is list:
            data_string = self.convert_list_to_string(data, indent=indent, eol=eol)
        elif type(data) is bool:
            data_string = str(data)
        elif type(data) is dict:
            data_string = self.convert_dictionary_to_string(data, indent=indent, separator=separator)
        elif type(data) is np.ndarray:
            # TensorBoard
            if format_type == 0:
                data_string = self.convert_ndarray_to_md(data)
            else:
                data_string = str(data)
        elif data is None:
            data_string = "None"
        else:
            if not self.ignore_unknown_dtypes:
                data_string = "Error:  MetaParameterRecorder Type conversion from type {} not supported.".\
                    format(str(type(data)))
                data_string += " (" + str(data) + ") "
            else:
                # TensorBoard
                if format_type == 0:
                    data_string = "**?**"

        return data_string

    def build_metagraph_list(self):
        """
        Convert MetaParams into TF Summary Format and create summary_op.

        Returns:
            Merged TF Op for TEXT summary elements, should only be executed once to reduce data duplication.

        """
        ops = []

        self.ignore_unknown_dtypes = True
        for key in sorted(self.meta_params):
            value = self.convert_data_to_string(self.meta_params[key])

            if len(value) == 0:
                continue
            if isinstance(value, str):
                ops.append(tf.contrib.summary.generic(name=key, tensor=tf.convert_to_tensor(str(value))))
            else:
                ops.append(tf.contrib.summary.generic(name=key, tensor=tf.as_string(tf.convert_to_tensor(value))))

        return ops
