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

from copy import deepcopy
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.debug import DumpingDebugWrapperSession

from tensorforce import TensorForceError, util
from tensorforce.core.explorations import Exploration
from tensorforce.core.preprocessors import PreprocessorStack


class Model(object):
    """
    The `Model` class coordinates the creation and execution of all TensorFlow operations within a model.
    It implements the `reset`, `act` and `update` functions, which form the interface the `Agent` class
    communicates with, and which should not need to be overwritten. Instead, the following TensorFlow
    functions need to be implemented:

    * `tf_actions_and_internals(states, internals, deterministic)` returning the batch of
       actions and successor internal states.

    Further, the following TensorFlow functions should be extended accordingly:

    * `setup_placeholders()` defining TensorFlow input placeholders for states, actions, rewards, etc..
    * `setup_template_funcs()` builds all TensorFlow functions from "tf_"-methods via tf.make_template.
    * `get_variables()` returning the list of TensorFlow variables (to be optimized) of this model.

    Finally, the following TensorFlow functions can be useful in some cases:

    * `tf_preprocess(states, internals, reward)` for states/action/reward preprocessing (e.g. reward normalization),
        returning the pre-processed tensors.
    * `tf_action_exploration(action, exploration, actions)` for action postprocessing (e.g. exploration),
        returning the processed batch of actions.
    * `create_output_operations(states, internals, actions, terminal, reward, deterministic)` for further output operations,
        similar to the two above for `Model.act` and `Model.update`.
    * `tf_optimization(states, internals, actions, terminal, reward)` for further optimization operations
        (e.g. the baseline update in a `PGModel` or the target network update in a `QModel`),
        returning a single grouped optimization operation.
    """

    def __init__(
        self,
        states,
        actions,
        scope,
        device,
        saver,
        summarizer,
        execution,
        batching_capacity,
        variable_noise,
        states_preprocessing,
        actions_exploration,
        reward_preprocessing,
        tf_session_dump_dir=""
    ):
        """
        Model.

        Args:
            states (spec): The state-space description dictionary.
            actions (spec): The action-space description dictionary.
            scope (str): The root scope str to use for tf variable scoping.
            device (str): The name of the device to run the graph of this model on.
            saver (spec): Dict specifying whether and how to save the model's parameters.
            summarizer (spec): Dict specifying which tensorboard summaries should be created and added to the graph.
            execution (spec): Dict specifying whether and how to do distributed training on the model's graph.
            batching_capacity (int): Batching capacity.
            variable_noise (float): The stddev value of a Normal distribution used for adding random
                noise to the model's output (for each batch, noise can be toggled and - if active - will be resampled).
                Use None for not adding any noise.
            states_preprocessing (spec / dict of specs): Dict specifying whether and how to preprocess state signals
                (e.g. normalization, greyscale, etc..).
            actions_exploration (spec / dict of specs): Dict specifying whether and how to add exploration to the model's
                "action outputs" (e.g. epsilon-greedy).
            reward_preprocessing (spec): Dict specifying whether and how to preprocess rewards coming
                from the Environment (e.g. reward normalization).
            tf_session_dump_dir (str): If non-empty string, all session.run calls will be dumped using the tensorflow
                offline-debug session into the given directory.
        """
        # Network crated from network in distribution_model.py
        # Needed for named_tensor access
        self.network = None

        # States/internals/actions specifications
        self.states_spec = states
        self.internals_spec = dict()
        self.actions_spec = actions

        # TensorFlow scope, device.
        self.scope = scope
        self.device = device

        # Saver/summaries
        if saver is None or saver.get('directory') is None:
            self.saver_spec = None
        else:
            self.saver_spec = saver
        if summarizer is None or summarizer.get('directory') is None:
            self.summarizer_spec = None
        else:
            self.summarizer_spec = summarizer
        self.summarizer = None  # The tf.summary.FileWriter object
        self.summarizer_hook = None  # The summarizer hook to use in the model's session.

        # TensorFlow summaries
        if self.summarizer_spec is None:
            self.summary_labels = set()
        else:
            self.summary_labels = set(self.summarizer_spec.get('labels', ()))
        self.is_observe = False

        # Execution logic settings.
        self.execution_spec = execution
        # Default single-process execution.
        self.execution_type = self.execution_spec["type"]
        self.session_config = self.execution_spec["session_config"]
        self.distributed_spec = self.execution_spec["distributed_spec"]

        # Batching capacity for act/observe interface
        assert batching_capacity is None or (isinstance(batching_capacity, int) and batching_capacity > 0)
        self.batching_capacity = batching_capacity or 1  # default

        # For offline debugging purposes. Off by default.
        self.tf_session_dump_dir = tf_session_dump_dir

        # Model's (tensorflow) buffers (states, actions, internals):
        # One record is inserted into these buffers when act(independent=False) method is called.
        self.states_buffer = dict()
        self.internals_buffer = dict()
        self.actions_buffer = dict()
        # Tensorflow int-index; reset to 0 when observe() is called.
        self.buffer_index = None
        # main-op executed in observe()
        self.episode_output = None
        self.unbuffered_episode_output = None

        # Variable noise
        assert variable_noise is None or variable_noise > 0.0
        self.variable_noise = variable_noise

        # Preprocessing and exploration
        self.states_preprocessing_spec = states_preprocessing
        self.actions_exploration_spec = actions_exploration
        self.reward_preprocessing_spec = reward_preprocessing

        self.variables = None
        self.all_variables = None
        self.registered_variables = None
        self.summaries = None

        # 0D counter tensors
        self.timestep = None
        self.episode = None
        self.global_timestep = None
        self.global_episode = None

        # input placeholders
        self.states_input = dict()
        self.states_preprocessing = dict()
        self.internals_input = dict()
        self.internals_init = dict()
        self.actions_input = dict()
        self.actions_exploration = dict()
        self.terminal_input = None
        self.reward_input = None
        self.reward_preprocessing = None
        self.deterministic_input = None
        self.independent_input = None
        self.update_input = None

        # template functions that return output tensors
        self.fn_initialize = None
        self.fn_preprocess = None
        self.fn_actions_and_internals = None
        self.fn_observe_timestep = None
        self.fn_action_exploration = None

        self.graph = None
        self.global_model = None
        # Whether this is a local replica of some global model (or we do single execution).
        self.is_local_model = True
        # The tf Server object (if any)
        self.server = None

        self.summary_writer = None
        self.summary_writer_hook = None
        self.saver = None
        self.saver_directory = None
        self.scaffold = None
        self.session = None
        self.monitored_session = None

        self.actions_output = None
        self.internals_output = None
        self.timestep_output = None

        self.summary_configuration_op = None

        # Setup Model (create and build graph (local and global if distributed), server, session, etc..).
        self.setup()

    def setup(self):
        """
        Sets up the TensorFlow model graph, starts the servers (distributed mode), creates summarizers
        and savers, initializes (and enters) the TensorFlow session.
        """

        # Create/get our graph, setup local model/global model links, set scope and device.
        graph_default_context = self.setup_graph()

        # Start a tf Server (in case of distributed setup). Only start once.
        if self.execution_type == "distributed" and not self.server and not self.is_local_model:
            self.start_server()

        # build the graph
        with tf.device(device_name_or_function=self.device):
            with tf.variable_scope(name_or_scope=self.scope, reuse=False):

                # Variables and summaries
                self.variables = dict()
                self.all_variables = dict()
                self.registered_variables = set()
                self.summaries = list()

                # Build the graph's placeholders, tf_functions, etc
                self.setup_placeholders()
                # Create model's "external" components.
                # Create tensorflow functions from "tf_"-methods.
                self.setup_components_and_tf_funcs()

                # Create core variables (timestep, episode counters, buffers for states/actions/internals).
                self.fn_initialize()

                # Input tensors
                states = util.map_tensors(fn=tf.identity, tensors=self.states_input)
                internals = util.map_tensors(fn=tf.identity, tensors=self.internals_input)
                actions = util.map_tensors(fn=tf.identity, tensors=self.actions_input)
                terminal = tf.identity(input=self.terminal_input)
                reward = tf.identity(input=self.reward_input)
                # Probably both deterministic and independent should be the same at some point.
                deterministic = tf.identity(input=self.deterministic_input)
                independent = tf.identity(input=self.independent_input)

                states, actions, reward = self.fn_preprocess(states=states, actions=actions, reward=reward)

                self.create_operations(
                    states=states,
                    internals=internals,
                    actions=actions,
                    terminal=terminal,
                    reward=reward,
                    deterministic=deterministic,
                    independent=independent
                )

                # Add all summaries specified in summary_labels
                if any(k in self.summary_labels for k in ['inputs', 'states']):
                    for name, state in states.items():
                        summary = tf.summary.histogram(name=(self.scope + '/inputs/states/' + name), values=state)
                        self.summaries.append(summary)
                if any(k in self.summary_labels for k in ['inputs', 'actions']):
                    for name, action in actions.items():
                        summary = tf.summary.histogram(name=(self.scope + '/inputs/actions/' + name), values=action)
                        self.summaries.append(summary)
                if any(k in self.summary_labels for k in ['inputs', 'rewards']):
                    summary = tf.summary.histogram(name=(self.scope + '/inputs/rewards'), values=reward)
                    self.summaries.append(summary)

        # If we are a global model -> return here.
        # Saving, syncing, finalizing graph, session is done by local replica model.
        if self.execution_type == "distributed" and not self.is_local_model:
            return

        # Saver/Summary -> Scaffold.
        self.setup_saver()
        summaries = self.get_summaries()
        if len(summaries) > 0:
            summary_op = tf.summary.merge(inputs=summaries)
        else:
            summary_op = None

        self.setup_scaffold(summary_op)

        # Create necessary hooks for the upcoming session.
        hooks = self.setup_summary_and_saver_hooks()
        # We are done constructing: Finalize our graph, create and enter the session.
        self.setup_session(self.server, hooks, graph_default_context)

    def setup_graph(self):
        """
        Creates our Graph and figures out, which shared/global model to hook up to.
        If we are in a global-model's setup procedure, we do not create
        a new graph (return None as the context). We will instead use the already existing local replica graph
        of the model.

        Returns: None or the graph's as_default()-context.
        """
        graph_default_context = None

        # Single (non-distributed) mode.
        if self.execution_type == "single":
            self.graph = tf.Graph()
            graph_default_context = self.graph.as_default()
            graph_default_context.__enter__()
            self.global_model = None

        # Distributed tf
        elif self.execution_type == "distributed":
            # Parameter-server -> Do not build any graph.
            if self.distributed_spec["job"] == "ps":
                return None

            # worker -> construct the global (main) model; the one hosted on the ps,
            elif self.distributed_spec["job"] == "worker":
                # The local replica model.
                if self.is_local_model:
                    graph = tf.Graph()
                    graph_default_context = graph.as_default()
                    graph_default_context.__enter__()

                    # Now that the graph is created and entered -> deepcopoy ourselves and setup global model first,
                    # then continue.
                    self.global_model = deepcopy(self)
                    # Switch on global construction/setup-mode for the pass to setup().
                    self.global_model.is_local_model = False
                    self.global_model.setup()

                    self.graph = graph
                    self.as_local_model()
                    self.scope += '-worker' + str(self.distributed_spec["task_index"])
                # The global_model (whose Variables are hosted by the ps).
                else:
                    self.graph = tf.get_default_graph()  # lives in the same graph as local model
                    self.global_model = None
                    self.device = tf.train.replica_device_setter(
                        # Place its Variables on the parameter server(s) (round robin).
                        #ps_device="/job:ps",  # default
                        # Train-ops for the global_model are hosted locally (on this worker's node).
                        worker_device=self.device,
                        cluster=self.distributed_spec["cluster_spec"]
                    )
                    # Tag all tensors as 'ps' without task idx so they are shared between all workers.
                    self.scope += '-ps'
            else:
                raise TensorForceError("Unsupported job type: {}!".format(self.distributed_spec["job"]))
        else:
            raise TensorForceError("Unsupported distributed type: {}!".format(self.distributed_spec["type"]))

        return graph_default_context

    def start_server(self):
        """
        Creates and stores a tf server (and optionally joins it if we are a parameter-server).
        Only relevant, if we are running in distributed mode.
        """
        self.server = tf.train.Server(
            server_or_cluster_def=self.distributed_spec["cluster_spec"],
            job_name=self.distributed_spec["job"],
            task_index=self.distributed_spec["task_index"],
            protocol=self.distributed_spec.get("protocol"),
            config=self.distributed_spec.get("session_config"),
            start=True
        )
        if self.distributed_spec["job"] == "ps":
            self.server.join()
            # This is unreachable?
            quit()

    def setup_placeholders(self):
        """
        Creates the TensorFlow placeholders, variables, ops and functions for this model.
        NOTE: Does not add the internal state placeholders and initialization values to the model yet as that requires
        the model's Network (if any) to be generated first.
        """

        # States
        for name, state in self.states_spec.items():
            self.states_input[name] = tf.placeholder(
                dtype=util.tf_dtype(state['type']),
                shape=(None,) + tuple(state['shape']),
                name=('state-' + name)
            )

        # States preprocessing
        if self.states_preprocessing_spec is None:
            for name, state in self.states_spec.items():
                state['unprocessed_shape'] = state['shape']
        elif not isinstance(self.states_preprocessing_spec, list) and \
                all(name in self.states_spec for name in self.states_preprocessing_spec):
            for name, state in self.states_spec.items():
                if name in self.states_preprocessing_spec:
                    preprocessing = PreprocessorStack.from_spec(
                        spec=self.states_preprocessing_spec[name],
                        kwargs=dict(shape=state['shape'])
                    )
                    state['unprocessed_shape'] = state['shape']
                    state['shape'] = preprocessing.processed_shape(shape=state['unprocessed_shape'])
                    self.states_preprocessing[name] = preprocessing
                else:
                    state['unprocessed_shape'] = state['shape']
        # Single preprocessor for all components of our state space
        elif "type" in self.states_preprocessing_spec:
            preprocessing = PreprocessorStack.from_spec(spec=self.states_preprocessing_spec)
            for name, state in self.states_spec.items():
                state['unprocessed_shape'] = state['shape']
                state['shape'] = preprocessing.processed_shape(shape=state['unprocessed_shape'])
                self.states_preprocessing[name] = preprocessing
        else:
            for name, state in self.states_spec.items():
                preprocessing = PreprocessorStack.from_spec(
                    spec=self.states_preprocessing_spec,
                    kwargs=dict(shape=state['shape'])
                )
                state['unprocessed_shape'] = state['shape']
                state['shape'] = preprocessing.processed_shape(shape=state['unprocessed_shape'])
                self.states_preprocessing[name] = preprocessing

        # Actions
        for name, action in self.actions_spec.items():
            self.actions_input[name] = tf.placeholder(
                dtype=util.tf_dtype(action['type']),
                shape=(None,) + tuple(action['shape']),
                name=('action-' + name)
            )

        # Actions exploration
        if self.actions_exploration_spec is None:
            pass
        elif all(name in self.actions_spec for name in self.actions_exploration_spec):
            for name, action in self.actions_spec.items():
                if name in self.actions_exploration:
                    self.actions_exploration[name] = Exploration.from_spec(spec=self.actions_exploration_spec[name])
        else:
            for name, action in self.actions_spec.items():
                self.actions_exploration[name] = Exploration.from_spec(spec=self.actions_exploration_spec)

        # Terminal
        self.terminal_input = tf.placeholder(dtype=util.tf_dtype('bool'), shape=(None,), name='terminal')

        # Reward
        self.reward_input = tf.placeholder(dtype=util.tf_dtype('float'), shape=(None,), name='reward')

        # Reward preprocessing
        if self.reward_preprocessing_spec is not None:
            self.reward_preprocessing = PreprocessorStack.from_spec(
                spec=self.reward_preprocessing_spec,
                # TODO this can eventually have more complex shapes?
                kwargs=dict(shape=())
            )
            if self.reward_preprocessing.processed_shape(shape=()) != ():
                raise TensorForceError("Invalid reward preprocessing!")

        # Deterministic/independent action flag (should probably be the same)
        self.deterministic_input = tf.placeholder(dtype=util.tf_dtype('bool'), shape=(), name='deterministic')
        self.independent_input = tf.placeholder(dtype=util.tf_dtype('bool'), shape=(), name='independent')

    def setup_components_and_tf_funcs(self, custom_getter=None):
        """
        Allows child models to create model's component objects, such as optimizer(s), memory(s), etc..
        Creates all tensorflow functions via tf.make_template calls on all the class' "tf_"-methods.

        Args:
            custom_getter: The `custom_getter_` object to use for `tf.make_template` when creating TensorFlow functions.
                If None, use a default custom_getter_.

        Returns: The custom_getter passed in (or a default one if custom_getter was None).
        """

        if custom_getter is None:
            def custom_getter(getter, name, registered=False, **kwargs):
                """
                To be passed to tf.make_template() as 'custom_getter_'.
                """
                if registered:
                    self.registered_variables.add(name)
                elif name in self.registered_variables:
                    registered = True
                # Top-level, hence no 'registered' argument.
                variable = getter(name=name, **kwargs)
                if not registered:
                    self.all_variables[name] = variable
                    if kwargs.get('trainable', True):
                        self.variables[name] = variable
                        if 'variables' in self.summary_labels:
                            summary = tf.summary.histogram(name=name, values=variable)
                            self.summaries.append(summary)
                return variable

        self.fn_initialize = tf.make_template(
            name_='initialize',
            func_=self.tf_initialize,
            custom_getter_=custom_getter
        )
        self.fn_preprocess = tf.make_template(
            name_='preprocess',
            func_=self.tf_preprocess,
            custom_getter_=custom_getter
        )
        self.fn_actions_and_internals = tf.make_template(
            name_='actions-and-internals',
            func_=self.tf_actions_and_internals,
            custom_getter_=custom_getter
        )
        self.fn_observe_timestep = tf.make_template(
            name_='observe-timestep',
            func_=self.tf_observe_timestep,
            custom_getter_=custom_getter
        )
        self.fn_action_exploration = tf.make_template(
            name_='action-exploration',
            func_=self.tf_action_exploration,
            custom_getter_=custom_getter
        )

        return custom_getter

    def setup_saver(self):
        """
        Creates the tf.train.Saver object and stores it in self.saver.
        """
        if self.execution_type == "single":
            global_variables = self.get_variables(include_submodules=True, include_nontrainable=True)
        else:
            global_variables = self.global_model.get_variables(include_submodules=True, include_nontrainable=True)

        # global_variables += [self.global_episode, self.global_timestep]

        for c in self.get_savable_components():
            c.register_saver_ops()

        # TensorFlow saver object
        # TODO potentially make other options configurable via saver spec.
        self.saver = tf.train.Saver(
            var_list=global_variables,  # should be given?
            reshape=False,
            sharded=False,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=10000.0,
            name=None,
            restore_sequentially=False,
            saver_def=None,
            builder=None,
            defer_build=False,
            allow_empty=True,
            write_version=tf.train.SaverDef.V2,
            pad_step_number=False,
            save_relative_paths=True
            # filename=None
        )

    def setup_scaffold(self, summary_op):
        """
        Creates the tf.train.Scaffold object with the given summary_op and assigns it to self.scaffold.
        Other fields of the Scaffold are generated automatically.
        """
        if self.execution_type == "single":
            global_variables = self.get_variables(include_submodules=True, include_nontrainable=True)
            #global_variables += [self.global_episode, self.global_timestep]
            init_op = tf.variables_initializer(var_list=global_variables)
            ready_op = tf.report_uninitialized_variables(var_list=global_variables)
            ready_for_local_init_op = None
            local_init_op = None

        else:
            # Global and local variable initializers.
            global_variables = self.global_model.get_variables(include_submodules=True, include_nontrainable=True)
            #global_variables += [self.global_episode, self.global_timestep]
            local_variables = self.get_variables(include_submodules=True, include_nontrainable=True)
            init_op = tf.variables_initializer(var_list=global_variables)
            ready_op = tf.report_uninitialized_variables(var_list=(global_variables + local_variables))
            ready_for_local_init_op = tf.report_uninitialized_variables(var_list=global_variables)
            local_init_op = tf.group(
                tf.variables_initializer(var_list=local_variables),
                # Synchronize values of trainable variables.
                *(tf.assign(ref=local_var, value=global_var) for local_var, global_var in zip(
                    self.get_variables(include_submodules=True),
                    self.global_model.get_variables(include_submodules=True)
                ))
            )

        def init_fn(scaffold, session):
            if self.saver_spec is not None and self.saver_spec.get('load', True):
                directory = self.saver_spec['directory']
                file = self.saver_spec.get('file')
                if file is None:
                    file = tf.train.latest_checkpoint(
                        checkpoint_dir=directory,
                        latest_filename=None  # Corresponds to argument of saver.save() in Model.save().
                    )
                elif not os.path.isfile(file):
                    file = os.path.join(directory, file)
                if file is not None:
                    scaffold.saver.restore(sess=session, save_path=file)

        # TensorFlow scaffold object
        # TODO explain what it does.
        self.scaffold = tf.train.Scaffold(
            init_op=init_op,
            init_feed_dict=None,
            init_fn=init_fn,
            ready_op=ready_op,
            ready_for_local_init_op=ready_for_local_init_op,
            local_init_op=local_init_op,
            summary_op=summary_op,
            saver=self.saver,
            copy_from_scaffold=None
        )

    def setup_summary_and_saver_hooks(self):
        """
        Creates and returns a list of saver and summarizer hooks to use in a session. Populates self.saver_directory,
        self.summarizer_hook and self.summarizer.

        Returns: List of hooks to use in a session.
        """
        hooks = list()

        # Checkpoint saver hook
        if self.saver_spec is not None and (self.execution_type == 'single' or self.distributed_spec['task_index'] == 0):
            self.saver_directory = self.saver_spec['directory']
            hooks.append(tf.train.CheckpointSaverHook(
                checkpoint_dir=self.saver_directory,
                save_secs=self.saver_spec.get('seconds', None if 'steps' in self.saver_spec else 600),
                save_steps=self.saver_spec.get('steps'),  # Either one or the other has to be set.
                saver=None,  # None since given via 'scaffold' argument.
                checkpoint_basename=self.saver_spec.get('basename', 'model.ckpt'),
                scaffold=self.scaffold,
                listeners=None
            ))
        else:
            self.saver_directory = None

        # Summary saver hook
        if self.summarizer_spec is None:
            self.summarizer_hook = None
        else:
            # TensorFlow summary writer object
            self.summarizer = tf.summary.FileWriter(
                logdir=self.summarizer_spec['directory'],
                graph=self.graph,
                max_queue=10,
                flush_secs=120,
                filename_suffix=None
            )
            self.summarizer_hook = util.UpdateSummarySaverHook(
                model=self,
                save_steps=self.summarizer_spec.get('steps'),  # Either one or the other has to be set.
                save_secs=self.summarizer_spec.get('seconds', None if 'steps' in self.summarizer_spec else 120),
                output_dir=None,  # None since given via 'summary_writer' argument.
                summary_writer=self.summarizer,
                scaffold=self.scaffold,
                summary_op=None  # None since given via 'scaffold' argument.
            )
            hooks.append(self.summarizer_hook)

        if self.summarizer_spec and 'meta_param_recorder_class' in self.summarizer_spec:
            self.summary_configuration_op = self.summarizer_spec['meta_param_recorder_class'].build_metagraph_list()

        # Stop at step hook
        # hooks.append(tf.train.StopAtStepHook(
        #     num_steps=???,  # This makes more sense, if load and continue training.
        #     last_step=None  # Either one or the other has to be set.
        # ))

        # # Step counter hook
        # hooks.append(tf.train.StepCounterHook(
        #     every_n_steps=counter_config.get('steps', 100),  # Either one or the other has to be set.
        #     every_n_secs=counter_config.get('secs'),  # Either one or the other has to be set.
        #     output_dir=None,  # None since given via 'summary_writer' argument.
        #     summary_writer=summary_writer
        # ))

        # Other available hooks:
        # tf.train.FinalOpsHook(final_ops, final_ops_feed_dict=None)
        # tf.train.GlobalStepWaiterHook(wait_until_step)
        # tf.train.LoggingTensorHook(tensors, every_n_iter=None, every_n_secs=None)
        # tf.train.NanTensorHook(loss_tensor, fail_on_nan_loss=True)
        # tf.train.ProfilerHook(save_steps=None, save_secs=None, output_dir='', show_dataflow=True, show_memory=False)

        return hooks

    def setup_session(self, server, hooks, graph_default_context):
        """
        Creates and then enters the session for this model (finalizes the graph).

        Args:
            server (tf.train.Server): The tf.train.Server object to connect to (None for single execution).
            hooks (list): A list of (saver, summary, etc..) hooks to be passed to the session.
            graph_default_context: The graph as_default() context that we are currently in.
        """
        if self.execution_type == "distributed":
            # if self.distributed_spec['task_index'] == 0:
            # TensorFlow chief session creator object
            session_creator = tf.train.ChiefSessionCreator(
                scaffold=self.scaffold,
                master=server.target,
                config=self.session_config,
                checkpoint_dir=None,
                checkpoint_filename_with_path=None
            )
            # else:
            #     # TensorFlow worker session creator object
            #     session_creator = tf.train.WorkerSessionCreator(
            #         scaffold=self.scaffold,
            #         master=server.target,
            #         config=self.execution_spec.get('session_config'),
            #     )

            # TensorFlow monitored session object
            self.monitored_session = tf.train.MonitoredSession(
                session_creator=session_creator,
                hooks=hooks,
                stop_grace_period_secs=120  # Default value.
            )
            # Add debug session.run dumping?
            if self.tf_session_dump_dir != "":
                self.monitored_session = DumpingDebugWrapperSession(self.monitored_session, self.tf_session_dump_dir)
        else:
            # TensorFlow non-distributed monitored session object
            self.monitored_session = tf.train.SingularMonitoredSession(
                hooks=hooks,
                scaffold=self.scaffold,
                master='',  # Default value.
                config=self.session_config,  # self.execution_spec.get('session_config'),
                checkpoint_dir=None
            )

        if graph_default_context:
            graph_default_context.__exit__(None, None, None)
        self.graph.finalize()

        # enter the session to be ready for acting/learning
        self.monitored_session.__enter__()
        self.session = self.monitored_session._tf_sess()

    def close(self):
        """
        Saves the model (of saver dir is given) and closes the session.
        """
        if self.saver_directory is not None:
            self.save(append_timestep=True)
        self.monitored_session.close()

    def as_local_model(self):
        pass

    def tf_initialize(self):
        """
        Creates tf Variables for the local state/internals/action-buffers and for the local and global counters
        for timestep and episode.
        """

        # Timesteps/Episodes
        # Global: (force on global device; local and global model point to the same (global) data).
        with tf.device(device_name_or_function=(self.global_model.device if self.global_model else self.device)):
            self.global_timestep = tf.get_variable(
                name='global-timestep',
                dtype=util.tf_dtype('int'),
                trainable=False,
                initializer=0,
                collections=['global-timestep', tf.GraphKeys.GLOBAL_STEP]
            )
            self.global_episode = tf.get_variable(
                name='global-episode',
                dtype=util.tf_dtype('int'),
                trainable=False,
                initializer=0,
                collections=['global-episode']
            )
        # Local counters: local device
        self.timestep = tf.get_variable(
            name='timestep',
            dtype=util.tf_dtype('int'),
            initializer=0,
            trainable=False
        )
        self.episode = tf.get_variable(
            name='episode',
            dtype=util.tf_dtype('int'),
            initializer=0,
            trainable=False
        )

        # States buffer variable
        for name, state in self.states_spec.items():
            self.states_buffer[name] = tf.get_variable(
                name=('state-' + name),
                shape=((self.batching_capacity,) + tuple(state['shape'])),
                dtype=util.tf_dtype(state['type']),
                trainable=False
            )

        # Internals buffer variable
        for name, internal in self.internals_spec.items():
            self.internals_buffer[name] = tf.get_variable(
                name=('internal-' + name),
                shape=((self.batching_capacity,) + tuple(internal['shape'])),
                dtype=util.tf_dtype(internal['type']),
                trainable=False
            )

        # Actions buffer variable
        for name, action in self.actions_spec.items():
            self.actions_buffer[name] = tf.get_variable(
                name=('action-' + name),
                shape=((self.batching_capacity,) + tuple(action['shape'])),
                dtype=util.tf_dtype(action['type']),
                trainable=False
            )

        # Buffer index
        self.buffer_index = tf.get_variable(
            name='buffer-index',
            shape=(),
            dtype=util.tf_dtype('int'),
            trainable=False
        )

    def tf_preprocess(self, states, actions, reward):
        """
        Applies preprocessing ops to the raw states/action/reward inputs.

        Args:
            states (dict): Dict of raw state tensors.
            actions (dict): Dict or raw action tensors.
            reward: 1D (float) raw rewards tensor.

        Returns: The preprocessed versions of the input tensors.
        """
        # States preprocessing
        for name, preprocessing in self.states_preprocessing.items():
            states[name] = preprocessing.process(tensor=states[name])

        # Reward preprocessing
        if self.reward_preprocessing is not None:
            reward = self.reward_preprocessing.process(tensor=reward)

        return states, actions, reward

    def tf_action_exploration(self, action, exploration, action_spec):
        """
        Applies optional exploration to the action (post-processor for action outputs).

        Args:
             action (tf.Tensor): The original output action tensor (to be post-processed).
             exploration (Exploration): The Exploration object to use.
             action_spec (dict): Dict specifying the action space.
        Returns:
            The post-processed action output tensor.
        """
        action_shape = tf.shape(input=action)
        exploration_value = exploration.tf_explore(
            episode=self.global_episode,
            timestep=self.global_timestep,
            action_spec=action_spec
        )

        if action_spec['type'] == 'bool':
            action = tf.where(
                condition=(tf.random_uniform(shape=action_shape[0]) < exploration_value),
                x=(tf.random_uniform(shape=action_shape) < 0.5),
                y=action
            )

        elif action_spec['type'] == 'int':
            action = tf.where(
                condition=(tf.random_uniform(shape=action_shape) < exploration_value),
                x=tf.random_uniform(shape=action_shape, maxval=action_spec['num_actions'], dtype=util.tf_dtype('int')),
                y=action
            )

        elif action_spec['type'] == 'float':
            for _ in range(util.rank(action) - 1):
                exploration_value = tf.expand_dims(input=exploration_value, axis=-1)
            action += exploration_value
            if 'min_value' in action_spec:
                action = tf.clip_by_value(
                    t=action,
                    clip_value_min=action_spec['min_value'],
                    clip_value_max=action_spec['max_value']
                )

        return action

    def tf_actions_and_internals(self, states, internals, deterministic):
        """
        Creates and returns the TensorFlow operations for retrieving the actions and - if applicable -
        the posterior internal state Tensors in reaction to the given input states (and prior internal states).

        Args:
            states (dict): Dict of state tensors (each key represents one state space component).
            internals (dict): Dict of internal state tensors (each key represents one internal space component).
            deterministic: Boolean tensor indicating whether action should be chosen
                deterministically.

        Returns:
            tuple:
                1) dict of output actions (with or without exploration applied (see `deterministic`))
                2) list of posterior internal state Tensors (empty for non-internal state models)
        """
        raise NotImplementedError

    def tf_observe_timestep(self, states, internals, actions, terminal, reward):
        """
        Creates the TensorFlow operations for processing a batch of observations coming in from our buffer (state,
        action, internals) as well as from the agent's python-batch (terminal-signals and rewards from the env).

        Args:
            states (dict): Dict of state tensors (each key represents one state space component).
            internals (dict): Dict of prior internal state tensors (each key represents one internal state component).
            actions (dict): Dict of action tensors (each key represents one action space component).
            terminal: 1D (bool) tensor of terminal signals.
            reward: 1D (float) tensor of rewards.

        Returns:
            The observation operation depending on the model type.
        """
        raise NotImplementedError

    def create_act_operations(self, states, internals, deterministic, independent):
        """
        Creates and stores tf operations that are fetched when calling act(): actions_output, internals_output and
        timestep_output.

        Args:
            states (dict): Dict of state tensors (each key represents one state space component).
            internals (dict): Dict of prior internal state tensors (each key represents one internal state component).
            deterministic: 0D (bool) tensor (whether to not use action exploration).
            independent (bool): 0D (bool) tensor (whether to store states/internals/action in local buffer).
        """

        # Optional variable noise
        operations = list()
        if self.variable_noise is not None and self.variable_noise > 0.0:
            # Initialize variables
            self.fn_actions_and_internals(
                states=states,
                internals=internals,
                deterministic=deterministic
            )

            noise_deltas = list()
            for variable in self.get_variables():
                noise_delta = tf.random_normal(shape=util.shape(variable), mean=0.0, stddev=self.variable_noise)
                noise_deltas.append(noise_delta)
                operations.append(variable.assign_add(delta=noise_delta))

        # Retrieve actions and internals
        with tf.control_dependencies(control_inputs=operations):
            self.actions_output, self.internals_output = self.fn_actions_and_internals(
                states=states,
                internals=internals,
                deterministic=deterministic
            )

        # Subtract variable noise
        # TODO this is an untested/incomplete feature and maybe should be removed for now.
        with tf.control_dependencies(control_inputs=list(self.actions_output.values())):
            operations = list()
            if self.variable_noise is not None and self.variable_noise > 0.0:
                for variable, noise_delta in zip(self.get_variables(), noise_deltas):
                    operations.append(variable.assign_sub(delta=noise_delta))

        # Actions exploration
        with tf.control_dependencies(control_inputs=operations):
            for name, exploration in self.actions_exploration.items():
                self.actions_output[name] = tf.cond(
                    pred=self.deterministic_input,
                    true_fn=(lambda: self.actions_output[name]),
                    false_fn=(lambda: self.fn_action_exploration(
                        action=self.actions_output[name],
                        exploration=exploration,
                        action_spec=self.actions_spec[name]
                    ))
                )

        # Independent act not followed by observe.
        def independent_act():
            """
            Does not store state, action, internal in buffer. Hence, does not have any influence on learning.
            Does not increase timesteps.
            """
            return self.global_timestep

        # Normal act followed by observe, with additional operations.
        def normal_act():
            """
            Stores current states, internals and actions in buffer. Increases timesteps.
            """
            operations = list()
            batch_size = tf.shape(input=next(iter(states.values())))[0]
            for name, state in states.items():
                operations.append(tf.assign(
                    ref=self.states_buffer[name][self.buffer_index: self.buffer_index + batch_size],
                    value=state
                ))
            for name, internal in internals.items():
                operations.append(tf.assign(
                    ref=self.internals_buffer[name][self.buffer_index: self.buffer_index + batch_size],
                    value=internal
                ))
            for name, action in self.actions_output.items():
                operations.append(tf.assign(
                    ref=self.actions_buffer[name][self.buffer_index: self.buffer_index + batch_size],
                    value=action
                ))

            with tf.control_dependencies(control_inputs=operations):
                operations = list()

                operations.append(tf.assign_add(ref=self.buffer_index, value=batch_size))

                # Increment timestep
                operations.append(tf.assign_add(ref=self.timestep, value=batch_size))
                operations.append(tf.assign_add(ref=self.global_timestep, value=batch_size))

            with tf.control_dependencies(control_inputs=operations):
                # Trivial operation to enforce control dependency
                # TODO why not return no-op?
                return self.global_timestep + 0

        # Only increment timestep and update buffer if act not independent
        self.timestep_output = tf.cond(
            pred=independent,
            true_fn=independent_act,
            false_fn=normal_act
        )

    def create_observe_operations(self, terminal, reward):
        """
        Returns the tf op to fetch when an observation batch is passed in (e.g. an episode's rewards and
        terminals). Uses the filled tf buffers for states, actions and internals to run
        the tf_observe_timestep (model-dependent), resets buffer index and increases counters (episodes,
        timesteps).

        Args:
            terminal: The 1D tensor (bool) of terminal signals to process (more than one True within that list is ok).
            reward: The 1D tensor (float) of rewards to process.

        Returns: Tf op to fetch when `observe()` is called.
        """
        # Increment episode
        num_episodes = tf.count_nonzero(input_tensor=terminal, dtype=util.tf_dtype('int'))
        increment_episode = tf.assign_add(ref=self.episode, value=num_episodes)
        increment_global_episode = tf.assign_add(ref=self.global_episode, value=num_episodes)

        with tf.control_dependencies(control_inputs=(increment_episode, increment_global_episode)):
            # Stop gradients
            fn = (lambda x: tf.stop_gradient(input=x[:self.buffer_index]))
            states = util.map_tensors(fn=fn, tensors=self.states_buffer)
            internals = util.map_tensors(fn=fn, tensors=self.internals_buffer)
            actions = util.map_tensors(fn=fn, tensors=self.actions_buffer)
            terminal = tf.stop_gradient(input=terminal)
            reward = tf.stop_gradient(input=reward)

            # Observation
            observation = self.fn_observe_timestep(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward
            )

        with tf.control_dependencies(control_inputs=(observation,)):
            # Reset buffer index.
            reset_index = tf.assign(ref=self.buffer_index, value=0)

        with tf.control_dependencies(control_inputs=(reset_index,)):
            # Trivial operation to enforce control dependency.
            self.episode_output = self.global_episode + 0

        # TODO: add up rewards per episode and add summary_label 'episode-reward'

    def create_atomic_observe_operations(self, states, actions, internals, terminal, reward):
        """
        Returns the tf op to fetch when unbuffered observations are passed in.

        Args:
            states (any): One state (usually a value tuple) or dict of states if multiple states are expected.
            actions (any): One action (usually a value tuple) or dict of states if multiple actions are expected.
            internals (any): Internal list.
            terminal (bool): boolean indicating if the episode terminated after the observation.
            reward (float): scalar reward that resulted from executing the action.

        Returns: Tf op to fetch when `observe()` is called.
        """
        # Increment episode
        num_episodes = tf.count_nonzero(input_tensor=terminal, dtype=util.tf_dtype('int'))
        increment_episode = tf.assign_add(ref=self.episode, value=num_episodes)
        increment_global_episode = tf.assign_add(ref=self.global_episode, value=num_episodes)

        with tf.control_dependencies(control_inputs=(increment_episode, increment_global_episode)):
            # Stop gradients
            # Not using buffers here.
            fn = (lambda x: tf.stop_gradient(input=x))
            states = util.map_tensors(fn=fn, tensors=states)
            internals = util.map_tensors(fn=fn, tensors=internals)
            actions = util.map_tensors(fn=fn, tensors=actions)
            terminal = tf.stop_gradient(input=terminal)
            reward = tf.stop_gradient(input=reward)

            # Observation
            observation = self.fn_observe_timestep(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward
            )

        with tf.control_dependencies(control_inputs=(observation,)):
            # Trivial operation to enforce control dependency.
            self.unbuffered_episode_output = self.global_episode + 0

    def create_operations(self, states, internals, actions, terminal, reward, deterministic, independent):
        """
        Creates and stores tf operations for when `act()` and `observe()` are called.
        """
        self.create_act_operations(
            states=states,
            internals=internals,
            deterministic=deterministic,
            independent=independent
        )
        self.create_observe_operations(reward=reward, terminal=terminal)
        self.create_atomic_observe_operations(
            states=states,
            actions=actions,
            internals=internals,
            reward=reward,
            terminal=terminal
        )

    def get_variables(self, include_submodules=False, include_nontrainable=False):
        """
        Returns the TensorFlow variables used by the model.

        Args:
            include_submodules: Includes variables of submodules (e.g. baseline, target network)  
                if true.
            include_nontrainable: Includes non-trainable variables if true.

        Returns:
            List of variables.
        """
        if include_nontrainable:
            model_variables = [self.all_variables[key] for key in sorted(self.all_variables)]

            states_preprocessing_variables = [
                variable for preprocessing in self.states_preprocessing.values()
                for variable in preprocessing.get_variables()
            ]
            model_variables += states_preprocessing_variables

            actions_exploration_variables = [
                variable for exploration in self.actions_exploration.values()
                for variable in exploration.get_variables()
            ]
            model_variables += actions_exploration_variables

            if self.reward_preprocessing is not None:
                reward_preprocessing_variables = self.reward_preprocessing.get_variables()
                model_variables += reward_preprocessing_variables

        else:
            model_variables = [self.variables[key] for key in sorted(self.variables)]

        return model_variables

    def get_summaries(self):
        """
        Returns the TensorFlow summaries reported by the model

        Returns:
            List of summaries
        """
        return self.summaries

    def reset(self):
        """
        Resets the model to its initial state on episode start. This should also reset all preprocessor(s).

        Returns:
            tuple:
                Current episode, timestep counter and the shallow-copied list of internal state initialization Tensors.
        """

        fetches = [self.global_episode, self.global_timestep]

        # Loop through all preprocessors and reset them as well.
        for preprocessing in self.states_preprocessing.values():
            fetch = preprocessing.reset()
            if fetch is not None:
                fetches.extend(fetch)

        # Get the updated episode and timestep counts.
        fetch_list = self.monitored_session.run(fetches=fetches)
        episode, timestep = fetch_list[:2]

        return episode, timestep, self.internals_init

    def get_feed_dict(
        self,
        states=None,
        internals=None,
        actions=None,
        terminal=None,
        reward=None,
        deterministic=None,
        independent=None
    ):
        """
        Returns the feed-dict for the model's acting and observing tf fetches.

        Args:
            states (dict): Dict of state values (each key represents one state space component).
            internals (dict): Dict of internal state values (each key represents one internal state component).
            actions (dict): Dict of actions (each key represents one action space component).
            terminal (List[bool]): List of is-terminal signals.
            reward (List[float]): List of reward signals.
            deterministic (bool): Whether actions should be picked without exploration.
            independent (bool): Whether we are doing an independent act (not followed by call to observe;
                not to be stored in model's buffer).

        Returns: The feed dict to use for the fetch.
        """
        feed_dict = dict()
        batched = None

        if states is not None:
            if batched is None:
                name = next(iter(states))
                state = np.asarray(states[name])
                batched = (state.ndim != len(self.states_spec[name]['unprocessed_shape']))
            if batched:
                feed_dict.update({state_input: states[name] for name, state_input in self.states_input.items()})
            else:
                feed_dict.update({state_input: (states[name],) for name, state_input in self.states_input.items()})

        if internals is not None:
            if batched is None:
                name = next(iter(internals))
                internal = np.asarray(internals[name])
                batched = (internal.ndim != len(self.internals_spec[name]['shape']))
            if batched:
                feed_dict.update({internal_input: internals[name] for name, internal_input in self.internals_input.items()})
            else:
                feed_dict.update({internal_input: (internals[name],) for name, internal_input in self.internals_input.items()})

        if actions is not None:
            if batched is None:
                name = next(iter(actions))
                action = np.asarray(actions[name])
                batched = (action.ndim != len(self.actions_spec[name]['shape']))
            if batched:
                feed_dict.update({action_input: actions[name] for name, action_input in self.actions_input.items()})
            else:
                feed_dict.update({action_input: (actions[name],) for name, action_input in self.actions_input.items()})

        if terminal is not None:
            if batched is None:
                terminal = np.asarray(terminal)
                batched = (terminal.ndim == 1)
            if batched:
                feed_dict[self.terminal_input] = terminal
            else:
                feed_dict[self.terminal_input] = (terminal,)

        if reward is not None:
            if batched is None:
                reward = np.asarray(reward)
                batched = (reward.ndim == 1)
            if batched:
                feed_dict[self.reward_input] = reward
            else:
                feed_dict[self.reward_input] = (reward,)

        if deterministic is not None:
            feed_dict[self.deterministic_input] = deterministic

        if independent is not None:
            feed_dict[self.independent_input] = independent

        return feed_dict

    def act(self, states, internals, deterministic=False, independent=False, fetch_tensors=None):
        """
        Does a forward pass through the model to retrieve action (outputs) given inputs for state (and internal
        state, if applicable (e.g. RNNs))

        Args:
            states (dict): Dict of state values (each key represents one state space component).
            internals (dict): Dict of internal state values (each key represents one internal state component).
            deterministic (bool): If True, will not apply exploration after actions are calculated.
            independent (bool): If true, action is not followed by observe (and hence not included
                in updates).
            fetch_tensors (list): List of names of additional tensors (from the model's network) to fetch (and return).

        Returns:
            tuple:
                - Actual action-outputs (batched if state input is a batch).
                - Actual values of internal states (if applicable) (batched if state input is a batch).
                - The timestep (int) after calculating the (batch of) action(s).
        """
        name = next(iter(states))
        state = np.asarray(states[name])
        batched = (state.ndim != len(self.states_spec[name]['unprocessed_shape']))
        if batched:
            assert state.shape[0] <= self.batching_capacity

        fetches = [self.actions_output, self.internals_output, self.timestep_output]
        if self.network is not None and fetch_tensors is not None:
            for name in fetch_tensors:
                valid, tensor = self.network.get_named_tensor(name)
                if valid:
                    fetches.append(tensor)
                else:
                    keys=self.network.get_list_of_named_tensor()
                    raise TensorForceError('Cannot fetch named tensor "{}", Available {}.'.format(name,keys))

        # feed_dict[self.deterministic_input] = deterministic
        feed_dict = self.get_feed_dict(
            states=states,
            internals=internals,
            deterministic=deterministic,
            independent=independent
        )

        fetch_list = self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)
        actions, internals, timestep = fetch_list[0:3]

        # Extract the first (and only) action/internal from the batch to make return values non-batched
        if not batched:
            actions = {name: action[0] for name, action in actions.items()}
            internals = {name: internal[0] for name, internal in internals.items()}

        if self.summary_configuration_op is not None:
            summary_values = self.session.run(self.summary_configuration_op)
            self.summarizer.add_summary(summary_values)
            self.summarizer.flush()
            # Only do this operation once to reduce duplicate data in Tensorboard
            self.summary_configuration_op = None

        if self.network is not None and fetch_tensors is not None:
            fetch_dict = dict()
            for index, tensor in enumerate(fetch_list[3:]):
                name = fetch_tensors[index]
                fetch_dict[name] = tensor
            return actions, internals, timestep, fetch_dict
        else:
            return actions, internals, timestep

    def observe(self, terminal, reward):
        """
        Adds an observation (reward and is-terminal) to the model without updating its trainable variables.

        Args:
            terminal (List[bool]): List of is-terminal signals.
            reward (List[float]): List of reward signals.

        Returns:
            The value of the model-internal episode counter.
        """

        fetches = self.episode_output
        feed_dict = self.get_feed_dict(terminal=terminal, reward=reward)

        self.is_observe = True  # for custom UpdateSummarySaverHook (see utils.py)
        episode = self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)
        self.is_observe = False

        return episode

    def atomic_observe(self, states, actions, internals, terminal, reward):
        fetches = self.unbuffered_episode_output
        feed_dict = self.get_feed_dict(
            states=states,
            actions=actions,
            internals=internals,
            terminal=terminal,
            reward=reward
        )

        self.is_observe = True
        episode = self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)
        self.is_observe = False

        return episode

    def save(self, directory=None, append_timestep=True):
        """
        Save TensorFlow model. If no checkpoint directory is given, the model's default saver  
        directory is used. Optionally appends current timestep to prevent overwriting previous  
        checkpoint files. Turn off to be able to load model from the same given path argument as  
        given here.

        Args:
            directory: Optional checkpoint directory.
            append_timestep: Appends the current timestep to the checkpoint file if true.

        Returns:
            Checkpoint path where the model was saved.
        """
        if self.summarizer_hook is not None:
            self.summarizer_hook._summary_writer.flush()

        return self.saver.save(
            sess=self.session,
            save_path=(self.saver_directory if directory is None else directory),
            global_step=(self.global_timestep if append_timestep else None),
            # latest_filename=None,  # Defaults to 'checkpoint'.
            meta_graph_suffix='meta',
            write_meta_graph=True,
            write_state=True
        )

    def restore(self, directory=None, file=None):
        """
        Restore TensorFlow model. If no checkpoint file is given, the latest checkpoint is  
        restored. If no checkpoint directory is given, the model's default saver directory is  
        used (unless file specifies the entire path).

        Args:
            directory: Optional checkpoint directory.
            file: Optional checkpoint file, or path if directory not given.
        """
        if file is None:
            file = tf.train.latest_checkpoint(
                checkpoint_dir=(self.saver_directory if directory is None else directory),
                # latest_filename=None  # Corresponds to argument of saver.save() in Model.save().
            )
        elif directory is None:
            file = os.path.join(self.saver_directory, file)
        elif not os.path.isfile(file):
            file = os.path.join(directory, file)

        # if not os.path.isfile(file):
        #     raise TensorForceError("Invalid model directory/file.")

        self.saver.restore(sess=self.session, save_path=file)

    def get_components(self):
        """
        Returns a dictionary of component name to component of all the components within this model.

        Returns:
            (dict) The mapping of name to component.
        """
        return dict()

    def get_savable_components(self):
        """
        Returns the list of all of the components this model consists of that can be individually saved and restored.
        For instance the network or distribution.

        Returns:
            List of util.SavableComponent
        """
        return set(filter(lambda x: isinstance(x, util.SavableComponent), self.get_components().values()))

    @staticmethod
    def _validate_savable(component, component_name):
        if not isinstance(component, util.SavableComponent):
            raise TensorForceError(
                "Component %s must implement SavableComponent but is %s" % (component_name, component)
            )

    def save_component(self, component_name, save_path):
        """
        Saves a component of this model to the designated location.

        Args:
            component_name: The component to save.
            save_path: The location to save to.
        Returns:
            Checkpoint path where the component was saved.
        """
        component = self.get_component(component_name=component_name)
        self._validate_savable(component=component, component_name=component_name)
        return component.save(sess=self.session, save_path=save_path)

    def restore_component(self, component_name, save_path):
        """
        Restores a component's parameters from a save location.

        Args:
            component_name: The component to restore.
            save_path: The save location.
        """
        component = self.get_component(component_name=component_name)
        self._validate_savable(component=component, component_name=component_name)
        component.restore(sess=self.session, save_path=save_path)

    def get_component(self, component_name):
        """
        Looks up a component by its name.

        Args:
            component_name: The name of the component to look up.
        Returns:
            The component for the provided name or None if there is no such component.
        """
        mapping = self.get_components()
        return mapping[component_name] if component_name in mapping else None
