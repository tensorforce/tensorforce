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
    Class to record MetaParameters as well as Summary/Description for TensorBoard (TEXT & FILE will come later)

    #### General:

    * format_type: used to configure data convertion for TensorBoard=0, TEXT & JSON (not Implemented), etc 
    """

    def __init__(self,current_frame):
        """
        Init the MetaPrameterRecord with "Agent" parameters by passing inspect.currentframe() from Agent Class
        
        The Init will search back to find the parent class to capture all passed parameters and store
        them in "self.meta_params".

        NOTE: Currently only optimized for TensorBoard output

        TODO: Add JSON Export, TEXT EXPORT

        Args:
            current_frame: frame value from class to obtain metaparameters[= inspect.currentframe()]

        """
        self.ignore_unknown_dtypes = False
        self.meta_params = dict()
        self.method_calling  = inspect.getframeinfo(current_frame)[2]

        _, _, __, self.vals_current =inspect.getargvalues(current_frame)
        if 'self' in self.vals_current:  #self is the class name of the frame involved
            self.recorded_class_type = self.vals_current['self']
            self.meta_params['AgentName'] = str(self.vals_current['self'])  # Add explicit AgentName item

        frame_list = inspect.getouterframes(current_frame)
        
        for frame in frame_list:           
            args, varargs, keywords, vals =inspect.getargvalues(frame[0])   #Rather than frame.frame (named tuple), use [0] for python2
            if 'self' in vals:
                if self.recorded_class_type == vals['self']:
                    for i in args:
                        self.meta_params[i] = vals[i]
        del self.meta_params['self']  # Remove the "CLASS" from the dictionary, has no value "AgentName" contains STR of Class
    

    def MergeCustom(self,custom_dict):
        if type(custom_dict) is not dict:
            raise TensorForceError("Error:  MetaParameterRecorder 'meta_dict' must be passed a dictionary but was passed a type {} which is not supported.".format(str(type(custom_dictdata))))
        for key in custom_dict:
            if key in self.meta_params:
                raise TensorForceError("Error:  MetaParameterRecorder 'meta_dict' key {} conflicts with internal key, please change passed key.".format(str(key)))
            self.meta_params[key] = custom_dict[key]

    def TextOutput(self,format_type=1):
        print('======================= '+self.meta_params['AgentName']+' ====================================')
        for key in self.meta_params:
            print ("    ",key,type(self.meta_params[key]),"=", self.ConvertDataToString(self.meta_params[key],format_type=format_type))      
        print('======================= '+self.meta_params['AgentName']+' ====================================')

    def ConvertDictionaryToString(self,data,indent=0,format_type=0,seperator=None,eol=None):
        data_string     = ""
        add_seperator   = ""
        if eol is None:
            eol = os.linesep        
        if seperator is None:
            seperator = ", "
        if type(data) is not dict: #This should not ever occur but here as a catch
            raise TensorForceError("Error:  MetaParameterRecorder Dictionary conversion was passed a type {} not supported.".format(str(type(data))))
        
        if format_type == 0: # TensorBoard
            label=""
            div=""
            if indent>0:
                label="    | "
                div  ="--- | "
            data_string += label + "Key | Value" + eol + div+ "--- | ----" +eol
        
        for key in data: # Should setup a TYPE here
            key_txt = key
            if format_type == 0: # TensorBoard
                key_txt = "**" + key + "**"
                key_value_sep = ' | '
                if indent>0:  # Add indent
                    key_txt="    | "+key_txt

            data_string += add_seperator+key_txt+key_value_sep+self.ConvertDataToString(data[key],seperator=seperator,indent=indent+1)  +eol
            #add_seperator = seperator
        
        return data_string  

    def ConvertListToString(self,data,indent=0,format_type=0,eol=None,count=True):
        data_string =""
        if eol is None:
            eol = os.linesep
        if type(data) is not list: #This should not ever occur but here as a catch
            raise TensorForceError("Error:  MetaParameterRecorder List conversion was passed a type {} not supported.".format(str(type(data))))

        for index,line in enumerate(data): # Should setup a TYPE here
            data_string_prefix = ""
            if count and indent==0:
                data_string_prefix = str(index+1)+". "
            if format_type == 0: # TensorBoard
                if indent>0 and index>0:  #Only add indent for 2nd item and beyond as this is likely a dictionary entry
                    data_string_prefix = "    | "+data_string_prefix
            if index==(len(data)-1):
                append_eol = ""  # don't append EOL
            else:
                append_eol = eol
            data_string += data_string_prefix + self.ConvertDataToString(line,indent=indent+1)+append_eol  
        
        return data_string  

    def ConvertNDArrayToMD(self,data,format_type=0,eol=None):
        data_string =""
        data_string1 = "|Row|"
        data_string2 = "|:---:|"
        if eol is None:
            eol = os.linesep
        if type(data) is not np.ndarray: #This should not ever occur but here as a catch
            raise TensorForceError("Error:  MetaParameterRecorder ndarray conversion was passed a type {} not supported.".format(str(type(data))))

        shape = data.shape
        rank = data.ndim

        if rank == 2:
            for col in range(shape[1]):
                data_string1 += "Col-"+str(col)+"|"
                data_string2 += ":----:|" 
            data_string += data_string1+eol+data_string2+eol

            for row in range(shape[0]):
                data_string +="|"+str(row)+"|"
                for col in range(shape[1]): 
                    data_string += str(data[row,col])+"|"

                if row!=(shape[0]-1):
                    data_string += eol

        elif rank == 1:
            data_string +="|Row|Col-0|"+eol+"|:----:|:----:|" + eol

            for row in range(shape[0]):
                data_string += str(row)+"|"+str(data[row]) +"|" + eol          

        return data_string      

    def ConvertDataToString(self,data,indent=0,format_type=0,seperator=None,eol=None):
        data_string =""
        if type(data) is int:
            data_string = str(data)
        elif type(data) is float:
            data_string = str(data)            
        elif type(data) is str:
            data_string = data      
        elif type(data) is tuple:
            data_string = str(data)                     
        elif type(data) is list:
            data_string = self.ConvertListToString(data,indent=indent,eol=eol) #str(data)  
        elif type(data) is bool:
            data_string = str(data)                        
        elif type(data) is dict:
            data_string = self.ConvertDictionaryToString(data,indent=indent,seperator=seperator)  #str(data)
        elif type(data) is np.ndarray:
            if format_type==0:  # TensorBoard
                data_string = self.ConvertNDArrayToMD(data)
            else:
                data_string = str(data)   
        elif data is None:
            data_string = "None"            
        else:
            if not self.ignore_unknown_dtypes:
                data_string ="Error:  MetaParameterRecorder Type conversion from type {} not supported.".format(str(type(data)))
                data_string += " ("+str(data)+") "
            else:
                if format_type == 0: # TensorBoard                
                    data_string = "**?**"
        
        return data_string

    def BuildMetaGraphList(self):  # returns list of summary ops
        ops = []

        self.ignore_unknown_dtypes = True
        for key in sorted(self.meta_params):
            value=self.ConvertDataToString(self.meta_params[key])

            if len(value) == 0:
                continue
            if isinstance(value,str):
                ops.append(tf.summary.text(key,tf.convert_to_tensor(str(value)  ))) #tf.convert_to_tensor(value)))
            else:
                ops.append(tf.summary.text(key, tf.as_string(tf.convert_to_tensor(value))))

        with tf.control_dependencies(tf.tuple(ops)):
            self.summary_merged = tf.summary.merge_all()

        return self.summary_merged

    