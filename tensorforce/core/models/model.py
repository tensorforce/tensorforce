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

from collections import OrderedDict
from copy import deepcopy
import os

import numpy as np
import tensorflow as tf

from tensorforce import TensorforceError, util
from tensorforce.core import Module, parameter_modules
from tensorforce.core.networks import Preprocessor


class Model(Module):

    def __init__(
        self,
        # Model
        name, device, parallel_interactions, buffer_observe, seed, execution, saver, summarizer,
        config, states, internals, actions, preprocessing, exploration, variable_noise,
        l2_regularization
    ):
        if summarizer is None or summarizer.get('directory') is None:
            summary_labels = None
        else:
            summary_labels = summarizer.get('labels', ('graph',))

        super().__init__(
            name=name, device=device, summary_labels=summary_labels,
            l2_regularization=l2_regularization
        )

        # Parallel interactions
        assert isinstance(parallel_interactions, int) and parallel_interactions >= 1
        self.parallel_interactions = parallel_interactions

        # Buffer observe
        assert isinstance(buffer_observe, int) and buffer_observe >= 1
        self.buffer_observe = buffer_observe

        # Seed
        self.seed = seed

        # Execution
        assert execution is None

        # Saver
        if saver is None:
            self.saver_spec = None
        elif not all(
            key in ('directory', 'filename', 'frequency', 'load', 'max-checkpoints')
            for key in saver
        ):
            raise TensorforceError.value(name='saver', value=list(saver))
        elif saver.get('directory') is None:
            self.saver_spec = None
        else:
            self.saver_spec = saver

        # Summarizer
        if summarizer is None:
            self.summarizer_spec = None
        elif not all(
            key in ('directory', 'flush', 'frequency', 'labels', 'max-summaries')
            for key in summarizer
        ):
            raise TensorforceError.value(name='summarizer', value=list(summarizer))
        elif summarizer.get('directory') is None:
            self.summarizer_spec = None
        else:
            self.summarizer_spec = summarizer

        self.config = config

        # States/internals/actions specifications
        self.states_spec = states
        self.internals_spec = OrderedDict() if internals is None else internals
        self.internals_init = OrderedDict()
        for name in self.internals_spec:
            self.internals_init[name] = None
            if name in self.states_spec:
                raise TensorforceError(
                    "Name overlap between internals and states: {}.".format(name)
                )
        self.actions_spec = actions
        for name in self.actions_spec:
            if name in self.states_spec:
                raise TensorforceError(
                    "Name overlap between actions and states: {}.".format(name)
                )
            if name in self.internals_spec:
                raise TensorforceError(
                    "Name overlap between actions and internals: {}.".format(name)
                )
        self.auxiliaries_spec = OrderedDict()
        for name, spec in self.actions_spec.items():
            if spec['type'] == 'int':
                name = name + '_mask'
                if name in self.states_spec:
                    raise TensorforceError(
                        "Name overlap between action-masks and states: {}.".format(name)
                    )
                if name in self.internals_spec:
                    raise TensorforceError(
                        "Name overlap between action-masks and internals: {}.".format(name)
                    )
                if name in self.actions_spec:
                    raise TensorforceError(
                        "Name overlap between action-masks and actions: {}.".format(name)
                    )
                self.auxiliaries_spec[name] = dict(
                    type='bool', shape=(spec['shape'] + (spec['num_values'],))
                )

        self.values_spec = OrderedDict(
            states=self.states_spec, internals=self.internals_spec,
            auxiliaries=self.auxiliaries_spec, actions=self.actions_spec,
            terminal=dict(type='long', shape=()), reward=dict(type='float', shape=())
        )

        # Preprocessing
        self.preprocessing = OrderedDict()
        self.unprocessed_state_shape = dict()
        for name, spec in self.states_spec.items():
            if preprocessing is None:
                layers = None
            elif name in preprocessing:
                layers = preprocessing[name]
            elif spec['type'] in preprocessing:
                layers = preprocessing[spec['type']]
            else:
                layers = None
            if layers is not None:
                self.unprocessed_state_shape[name] = spec['shape']
                self.preprocessing[name] = self.add_module(
                    name=(name + '-preprocessing'), module=Preprocessor, input_spec=spec,
                    layers=layers
                )
                self.states_spec[name] = self.preprocessing[name].get_output_spec()
        if preprocessing is not None and 'reward' in preprocessing:
            reward_spec = dict(type='float', shape=())
            self.preprocessing['reward'] = self.add_module(
                name=('reward-preprocessing'), module=Preprocessor, input_spec=reward_spec,
                layers=preprocessing['reward']
            )
            if self.preprocessing['reward'].get_output_spec() != reward_spec:
                raise TensorforceError.unexpected()

        # Exploration
        exploration = 0.0 if exploration is None else exploration
        if isinstance(exploration, dict) and \
                all(name in self.actions_spec for name in exploration):
            # Different exploration per action
            self.exploration = OrderedDict()
            for name in self.actions_spec:
                if name in exploration:
                    self.exploration[name] = self.add_module(
                        name=(name + '-exploration'), module=exploration[name],
                        modules=parameter_modules, is_trainable=False, dtype='float'
                    )
        else:
            # Same exploration for all actions
            self.exploration = self.add_module(
                name='exploration', module=exploration, modules=parameter_modules,
                is_trainable=False, dtype='float'
            )

        # Variable noise
        assert variable_noise is None or isinstance(variable_noise, dict) or variable_noise >= 0.0
        variable_noise = 0.0 if variable_noise is None else variable_noise
        self.variable_noise = self.add_module(
            name='variable-noise', module=variable_noise, modules=parameter_modules,
            is_trainable=False, dtype='float'
        )

        # Execution
        self.execution_spec = None
        if self.execution_spec is not None:
            self.execution_type = self.execution_spec['type']
            self.session_config = self.execution_spec['session_config']
            self.distributed_spec = self.execution_spec['distributed_spec']
        # One record is inserted into these buffers when act(independent=False) method is called.
        # self.num_parallel = self.execution_spec.get('num_parallel', 1)

        # Register global tensors
        for name, spec in self.states_spec.items():
            Module.register_tensor(name=name, spec=spec, batched=True)
        for name, spec in self.internals_spec.items():
            Module.register_tensor(name=name, spec=spec, batched=True)
        for name, spec in self.actions_spec.items():
            Module.register_tensor(name=name, spec=spec, batched=True)
        Module.register_tensor(name='terminal', spec=dict(type='long', shape=()), batched=True)
        Module.register_tensor(name='reward', spec=dict(type='float', shape=()), batched=True)
        Module.register_tensor(
            name='deterministic', spec=dict(type='bool', shape=()), batched=False
        )
        Module.register_tensor(name='independent', spec=dict(type='bool', shape=()), batched=False)
        Module.register_tensor(
            name='optimization', spec=dict(type='bool', shape=()), batched=False
        )
        Module.register_tensor(name='timestep', spec=dict(type='long', shape=()), batched=False)
        Module.register_tensor(name='episode', spec=dict(type='long', shape=()), batched=False)
        Module.register_tensor(name='update', spec=dict(type='long', shape=()), batched=False)

    def initialize(self):
        """
        Sets up the TensorFlow model graph, starts the servers (distributed mode), creates summarizers
        and savers, initializes (and enters) the TensorFlow session.
        """

        # Create/get our graph, setup local model/global model links, set scope and device.
        graph_default_context = self.setup_graph()

        # Start a tf Server (in case of distributed setup). Only start once.
        # if self.execution_type == "distributed" and self.server is None and self.is_local_model:
        if self.execution_spec is None or self.execution_type == 'local' or not self.is_local_model:
            self.server = None
        else:
            # Creates and stores a tf server (and optionally joins it if we are a parameter-server).
            # Only relevant, if we are running in distributed mode.
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

        super().initialize()

        # If we are a global model -> return here.
        # Saving, syncing, finalizing graph, session is done by local replica model.
        if self.execution_spec is not None and self.execution_type == "distributed" and not self.is_local_model:
            return

        # Saver/Summary -> Scaffold.
        # Creates the tf.train.Saver object and stores it in self.saver.
        if self.execution_spec is None or self.execution_type == "single":
            saved_variables = self.get_variables(only_saved=True)
        else:
            saved_variables = self.global_model.get_variables(only_saved=True)

        # global_variables += [self.global_episode, self.global_timestep]

        # for c in self.get_savable_components():
        #     c.register_saver_ops()

        # TensorFlow saver object
        # TODO potentially make other options configurable via saver spec.

        # possibility to turn off?
        if self.saver_spec is None:
            max_to_keep = 5
        else:
            max_to_keep = self.saver_spec.get('max-checkpoints', 5)
        self.saver = tf.train.Saver(
            var_list=saved_variables,  # should be given?
            reshape=False,
            sharded=False,
            max_to_keep=max_to_keep,
            keep_checkpoint_every_n_hours=10000.0,
            name=None,
            restore_sequentially=False,
            saver_def=None,
            builder=None,
            defer_build=False,
            allow_empty=False,
            write_version=tf.train.SaverDef.V2,
            pad_step_number=False,
            save_relative_paths=False,
            filename=None
        )

        self.setup_scaffold()

        # Create necessary hooks for the upcoming session.
        hooks = self.setup_hooks()

        # We are done constructing: Finalize our graph, create and enter the session.
        self.setup_session(self.server, hooks, graph_default_context)

        if self.saver_directory is not None:
            self.save()

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
        if self.execution_spec is None or self.execution_type == 'single':
            self.graph = tf.Graph()
            graph_default_context = self.graph.as_default()
            graph_default_context.__enter__()
            self.global_model = None

        # Distributed tf
        elif self.execution_type == 'distributed':
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

                    # self.as_local_model() for all optimizers:
                    # self.optimizer_spec = dict(
                    #     type='global_optimizer',
                    #     optimizer=self.optimizer_spec
                    # )
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
            else:
                raise TensorforceError("Unsupported job type: {}!".format(self.distributed_spec["job"]))
        else:
            raise TensorforceError("Unsupported distributed type: {}!".format(self.distributed_spec["type"]))

        if self.seed is not None:
            tf.random.set_random_seed(seed=self.seed)

        return graph_default_context

    def setup_scaffold(self):
        """
        Creates the tf.train.Scaffold object and assigns it to self.scaffold.
        Other fields of the Scaffold are generated automatically.
        """
        if self.execution_spec is None or self.execution_type == "single":
            global_variables = self.get_variables()
            # global_variables += [self.global_episode, self.global_timestep]
            init_op = tf.variables_initializer(var_list=global_variables)
            if self.summarizer_spec is not None:
                init_op = tf.group(init_op, self.summarizer_init)
            if self.graph_summary is None:
                ready_op = tf.report_uninitialized_variables(var_list=global_variables)
                ready_for_local_init_op = None
                local_init_op = None
            else:
                ready_op = None
                ready_for_local_init_op = tf.report_uninitialized_variables(var_list=global_variables)
                local_init_op = self.graph_summary

        else:
            # Global and local variable initializers.
            global_variables = self.global_model.get_variables()
            # global_variables += [self.global_episode, self.global_timestep]
            local_variables = self.get_variables()
            init_op = tf.variables_initializer(var_list=global_variables)
            if self.summarizer_spec is not None:
                init_op = tf.group(init_op, self.summarizer_init)
            ready_op = tf.report_uninitialized_variables(var_list=(global_variables + local_variables))
            ready_for_local_init_op = tf.report_uninitialized_variables(var_list=global_variables)
            if self.graph_summary is None:
                local_init_op = tf.group(
                    tf.variables_initializer(var_list=local_variables),
                    # Synchronize values of trainable variables.
                    *(tf.assign(ref=local_var, value=global_var) for local_var, global_var in zip(
                        self.get_variables(only_trainable=True),
                        self.global_model.get_variables(only_trainable=True)
                    ))
                )
            else:
                local_init_op = tf.group(
                    tf.variables_initializer(var_list=local_variables),
                    self.graph_summary,
                    # Synchronize values of trainable variables.
                    *(tf.assign(ref=local_var, value=global_var) for local_var, global_var in zip(
                        self.get_variables(only_trainable=True),
                        self.global_model.get_variables(only_trainable=True)
                    ))
                )

        def init_fn(scaffold, session):
            if self.saver_spec is not None and self.saver_spec.get('load', True):
                directory = self.saver_spec['directory']
                load = self.saver_spec.get('load')
                if isinstance(load, str):
                    save_path = os.path.join(directory, load)
                else:
                    save_path = tf.train.latest_checkpoint(
                        checkpoint_dir=directory, latest_filename=None
                    )
                if save_path is not None:
                    # global vs local model restored correctly?
                    scaffold.saver.restore(sess=session, save_path=save_path)
                    session.run(fetches=util.join_scopes(self.name + '.reset', 'timestep-output:0'))

        # TensorFlow scaffold object
        # TODO explain what it does.
        self.scaffold = tf.train.Scaffold(
            init_op=init_op,
            init_feed_dict=None,
            init_fn=init_fn,
            ready_op=ready_op,
            ready_for_local_init_op=ready_for_local_init_op,
            local_init_op=local_init_op,
            summary_op=None,
            saver=self.saver,
            copy_from_scaffold=None
        )

    def setup_hooks(self):
        """
        Creates and returns a list of hooks to use in a session. Populates self.saver_directory.

        Returns: List of hooks to use in a session.
        """
        hooks = list()

        # Checkpoint saver hook
        if self.saver_spec is not None:  # and (self.execution_type == 'single' or self.distributed_spec['task_index'] == 0):
            self.saver_directory = self.saver_spec['directory']
            self.saver_filename = self.saver_spec.get('filename', 'agent')
            frequency = self.saver_spec.get('frequency', 600)
            if frequency is not None:
                hooks.append(tf.train.CheckpointSaverHook(
                    checkpoint_dir=self.saver_directory, save_secs=frequency, save_steps=None,
                    saver=None,  # None since given via 'scaffold' argument.
                    checkpoint_basename=self.saver_filename, scaffold=self.scaffold, listeners=None
                ))
        else:
            self.saver_directory = None
            self.saver_filename = 'agent'

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
        if self.execution_spec is not None and self.execution_type == "distributed":
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
            # from tensorflow.python.debug import DumpingDebugWrapperSession
            # self.monitored_session = DumpingDebugWrapperSession(self.monitored_session, self.tf_session_dump_dir)

        else:
            # TensorFlow non-distributed monitored session object
            self.monitored_session = tf.train.SingularMonitoredSession(
                hooks=hooks,
                scaffold=self.scaffold,
                master='',  # Default value.
                config=(None if self.execution_spec is None else self.session_config),  # self.execution_spec.get('session_config'),
                checkpoint_dir=None
            )

        if graph_default_context:
            graph_default_context.__exit__(None, None, None)
        self.graph.finalize()

        # enter the session to be ready for acting/learning
        self.monitored_session.__enter__()
        self.session = self.monitored_session._tf_sess()

    def close(self):
        if self.summarizer_spec is not None:
            self.monitored_session.run(fetches=self.summarizer_close)
        if self.saver_directory is not None:
            self.save()
        self.monitored_session.__exit__(None, None, None)

    def tf_initialize(self):
        super().tf_initialize()

        # States
        self.states_input = OrderedDict()
        for name, state_spec in self.states_spec.items():
            self.states_input[name] = self.add_placeholder(
                name=name, dtype=state_spec['type'],
                shape=self.unprocessed_state_shape.get(name, state_spec['shape']), batched=True
            )

        # Auxiliaries
        self.auxiliaries_input = OrderedDict()
        # Categorical action masks
        for name, action_spec in self.actions_spec.items():
            if action_spec['type'] == 'int':
                name = name + '_mask'
                shape = action_spec['shape'] + (action_spec['num_values'],)
                default = tf.constant(
                    value=True, dtype=util.tf_dtype(dtype='bool'), shape=((1,) + shape)
                )
                self.auxiliaries_input[name] = self.add_placeholder(
                    name=name, dtype='bool', shape=shape, batched=True, default=default
                )

        # Terminal  (default: False?)
        self.terminal_input = self.add_placeholder(
            name='terminal', dtype='long', shape=(), batched=True
        )

        # Reward  (default: 0.0?)
        self.reward_input = self.add_placeholder(
            name='reward', dtype='float', shape=(), batched=True
        )

        # Deterministic flag
        self.deterministic_input = self.add_placeholder(
            name='deterministic', dtype='bool', shape=(), batched=False,
            default=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))
        )

        # Independent flag
        self.independent_input = self.add_placeholder(
            name='independent', dtype='bool', shape=(), batched=False,
            default=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))
        )

        # Parallel index
        self.parallel_input = self.add_placeholder(
            name='parallel', dtype='long', shape=(), batched=False,
            default=tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
        )

        # Local timestep
        self.timestep = self.add_variable(
            name='timestep', dtype='long', shape=(self.parallel_interactions,),
            initializer='zeros', is_trainable=False
        )

        # Local episode
        self.episode = self.add_variable(
            name='episode', dtype='long', shape=(self.parallel_interactions,), initializer='zeros',
            is_trainable=False
        )

        # Episode reward
        self.episode_reward = self.add_variable(
            name='episode-reward', dtype='float', shape=(self.parallel_interactions,),
            initializer='zeros', is_trainable=False
        )

        # States buffer variable
        self.states_buffer = OrderedDict()
        for name, spec in self.states_spec.items():
            self.states_buffer[name] = self.add_variable(
                name=(name + '-buffer'), dtype=spec['type'],
                shape=((self.parallel_interactions, self.buffer_observe) + spec['shape']),
                is_trainable=False, is_saved=False
            )

        # Internals buffer variable
        self.internals_buffer = OrderedDict()
        for name, spec in self.internals_spec.items():
            shape = ((self.parallel_interactions, self.buffer_observe + 1) + spec['shape'])
            initializer = np.zeros(shape=shape, dtype=util.np_dtype(dtype=spec['type']))
            initializer[:, 0] = self.internals_init[name]
            self.internals_buffer[name] = self.add_variable(
                name=(name + '-buffer'), dtype=spec['type'], shape=shape, is_trainable=False,
                initializer=initializer, is_saved=False
            )

        # Auxiliaries buffer variable
        self.auxiliaries_buffer = OrderedDict()
        for name, spec in self.auxiliaries_spec.items():
            self.auxiliaries_buffer[name] = self.add_variable(
                name=(name + '-buffer'), dtype=spec['type'],
                shape=((self.parallel_interactions, self.buffer_observe) + spec['shape']),
                is_trainable=False, is_saved=False
            )

        # Actions buffer variable
        self.actions_buffer = OrderedDict()
        for name, spec in self.actions_spec.items():
            self.actions_buffer[name] = self.add_variable(
                name=(name + '-buffer'), dtype=spec['type'],
                shape=((self.parallel_interactions, self.buffer_observe) + spec['shape']),
                is_trainable=False, is_saved=False
            )

        # Buffer index
        self.buffer_index = self.add_variable(
            name='buffer-index', dtype='long', shape=(self.parallel_interactions,),
            initializer='zeros', is_trainable=False, is_saved=False
        )

    def api_reset(self):
        assignment = self.buffer_index.assign(
            value=tf.zeros(shape=(self.parallel_interactions,), dtype=util.tf_dtype(dtype='long')),
            read_value=False
        )

        # Synchronization initial sync?

        with tf.control_dependencies(control_inputs=(assignment,)):
            timestep = util.identity_operation(
                x=self.global_timestep, operation_name='timestep-output'
            )
            episode = util.identity_operation(
                x=self.global_episode, operation_name='episode-output'
            )
            update = util.identity_operation(
                x=self.global_update, operation_name='update-output'
            )

        return timestep, episode, update

    def api_act(self):
        # Inputs
        states = self.states_input
        auxiliaries = self.auxiliaries_input
        parallel = self.parallel_input
        deterministic = self.deterministic_input
        independent = self.independent_input

        true = tf.constant(value=True, dtype=util.tf_dtype(dtype='bool'))
        zero_float = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))

        # Assertions
        assertions = list()
        # states: type and shape
        for name, spec in self.states_spec.items():
            assertions.append(
                tf.debugging.assert_type(
                    tensor=states[name], tf_type=util.tf_dtype(dtype=spec['type'])
                )
            )
            shape = (1,) + self.unprocessed_state_shape.get(name, spec['shape'])
            assertions.append(
                tf.debugging.assert_equal(
                    x=tf.shape(input=states[name], out_type=tf.int32),
                    y=tf.constant(value=shape, dtype=tf.int32)
                )
            )
        # action_masks: type and shape
        for name, spec in self.actions_spec.items():
            if spec['type'] == 'int':
                name = name + '_mask'
                assertions.append(
                    tf.debugging.assert_type(
                        tensor=auxiliaries[name], tf_type=util.tf_dtype(dtype='bool')
                    )
                )
                shape = (1,) + spec['shape'] + (spec['num_values'],)
                assertions.append(
                    tf.debugging.assert_equal(
                        x=tf.shape(input=auxiliaries[name], out_type=tf.int32),
                        y=tf.constant(value=shape, dtype=tf.int32)
                    )
                )
                assertions.append(
                    tf.debugging.assert_equal(
                        x=tf.reduce_all(
                            input_tensor=tf.reduce_any(
                                input_tensor=auxiliaries[name], axis=tuple(range(1, len(shape)))
                            ), axis=0
                        ), y=true
                    )
                )
        # parallel: type, shape and value
        assertions.append(
            tf.debugging.assert_type(tensor=parallel, tf_type=util.tf_dtype(dtype='long'))
        )
        assertions.append(tf.debugging.assert_scalar(tensor=parallel))
        assertions.append(tf.debugging.assert_non_negative(x=parallel))
        assertions.append(
            tf.debugging.assert_less(
                x=parallel,
                y=tf.constant(value=self.parallel_interactions, dtype=util.tf_dtype(dtype='long'))
            )
        )
        # deterministic: type and shape
        assertions.append(
            tf.debugging.assert_type(tensor=deterministic, tf_type=util.tf_dtype(dtype='bool'))
        )
        assertions.append(tf.debugging.assert_scalar(tensor=deterministic))
        # independent: type and shape
        assertions.append(
            tf.debugging.assert_type(tensor=independent, tf_type=util.tf_dtype(dtype='bool'))
        )
        assertions.append(tf.debugging.assert_scalar(tensor=independent))

        # Set global tensors
        Module.update_tensors(
            deterministic=deterministic, independent=independent,
            optimization=tf.constant(value=False, dtype=util.tf_dtype(dtype='bool')),
            timestep=self.global_timestep, episode=self.global_episode, update=self.global_update
        )

        one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))

        # Increment timestep
        with tf.control_dependencies(control_inputs=assertions):

            def increment_timestep():
                assignments = list()
                assignments.append(
                    self.timestep.scatter_nd_add(indices=[(parallel,)], updates=[one])
                )
                assignments.append(self.global_timestep.assign_add(delta=one, read_value=False))
                with tf.control_dependencies(control_inputs=assignments):
                    return util.no_operation()

            incremented_timestep = self.cond(
                pred=independent, true_fn=util.no_operation, false_fn=increment_timestep
            )
            dependencies = (incremented_timestep,)

        # Preprocessing states
        if any(name in self.preprocessing for name in self.states_spec):
            with tf.control_dependencies(control_inputs=dependencies):
                for name in self.states_spec:
                    if name in self.preprocessing:
                        states[name] = self.preprocessing[name].apply(x=states[name])
            dependencies = util.flatten(xs=states)

        # Variable noise
        variables = self.get_variables(only_trainable=True)
        if len(variables) > 0:
            with tf.control_dependencies(control_inputs=dependencies):
                variable_noise = self.variable_noise.value()

                def no_variable_noise():
                    noise_tensors = list()
                    for variable in variables:
                        noise_tensors.append(tf.zeros_like(tensor=variable, dtype=variable.dtype))
                    return noise_tensors

                def apply_variable_noise():
                    assignments = list()
                    noise_tensors = list()
                    for variable in variables:
                        if variable.dtype == util.tf_dtype(dtype='float'):
                            noise = tf.random.normal(
                                shape=util.shape(variable), mean=0.0, stddev=variable_noise,
                                dtype=util.tf_dtype(dtype='float')
                            )
                        else:
                            noise = tf.random.normal(
                                shape=util.shape(variable), mean=0.0,
                                stddev=tf.dtypes.cast(x=variable_noise, dtype=variable.dtype),
                                dtype=variable.dtype
                            )
                        noise_tensors.append(noise)
                        assignments.append(variable.assign_add(delta=noise, read_value=False))
                    with tf.control_dependencies(control_inputs=assignments):
                        return util.fmap(function=util.identity_operation, xs=noise_tensors)

                skip_variable_noise = tf.math.logical_or(
                    x=deterministic, y=tf.math.equal(x=variable_noise, y=zero_float)
                )
                variable_noise_tensors = self.cond(
                    pred=skip_variable_noise, true_fn=no_variable_noise,
                    false_fn=apply_variable_noise
                )
                dependencies = variable_noise_tensors

        # Initialize or retrieve internals
        if len(self.internals_spec) > 0:
            with tf.control_dependencies(control_inputs=dependencies):
                # buffer_index = self.buffer_index[parallel]

                # def initialize_internals():
                #     internals = OrderedDict()
                #     for name, init in self.internals_init.items():
                #         internals[name] = tf.expand_dims(input=init, axis=0)
                #     return internals

                # def retrieve_internals():
                #     internals = OrderedDict()
                #     for name in self.internals_spec:
                #         internals[name] = tf.gather_nd(
                #             params=self.internals_buffer[name],
                #             indices=[(parallel, buffer_index)]
                #         )
                #     return internals

                # zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))
                # initialize = tf.math.equal(x=buffer_index, y=zero)
                # internals = self.cond(
                #     pred=initialize, true_fn=initialize_internals, false_fn=retrieve_internals
                # )
                # retrieved_internals = util.flatten(xs=internals)
                # dependencies = retrieved_internals

                buffer_index = self.buffer_index[parallel]
                internals = OrderedDict()
                for name in self.internals_spec:
                    internals[name] = tf.gather_nd(
                        params=self.internals_buffer[name],
                        indices=[(parallel, buffer_index)]
                    )
                dependencies = util.flatten(xs=internals)
        else:
            internals = OrderedDict()

        # Core act: retrieve actions and internals
        with tf.control_dependencies(control_inputs=dependencies):
            actions, internals = self.core_act(
                states=states, internals=internals, auxiliaries=auxiliaries
            )
            dependencies = util.flatten(xs=actions) + util.flatten(xs=internals)

        # Check action masks
        # TODO: also check float bounds, move after exploration?
        assertions = list()
        for name, spec in self.actions_spec.items():
            if spec['type'] == 'int':
                indices = tf.dtypes.cast(x=actions[name], dtype=tf.int64)
                indices = tf.expand_dims(input=indices, axis=-1)
                is_unmasked = tf.batch_gather(params=auxiliaries[name + '_mask'], indices=indices)
                assertions.append(tf.debugging.assert_equal(
                    x=tf.math.reduce_all(input_tensor=is_unmasked), y=true
                ))
        dependencies += assertions

        # Exploration
        with tf.control_dependencies(control_inputs=dependencies):
            if not isinstance(self.exploration, dict):
                exploration = self.exploration.value()

            for name, spec in self.actions_spec.items():
                if isinstance(self.exploration, dict):
                    if name in self.exploration:
                        exploration = self.exploration[name].value()
                    else:
                        continue

                def no_exploration():
                    return actions[name]

                float_dtype = util.tf_dtype(dtype='float')
                shape = tf.shape(input=actions[name])

                if spec['type'] == 'bool':

                    def apply_exploration():
                        condition = tf.random.uniform(shape=shape, dtype=float_dtype) < exploration
                        half = tf.constant(value=0.5, dtype=float_dtype)
                        random_action = tf.random.uniform(shape=shape, dtype=float_dtype) < half
                        return tf.where(condition=condition, x=random_action, y=actions[name])

                elif spec['type'] == 'int':
                    int_dtype = util.tf_dtype(dtype='int')

                    def apply_exploration():
                        # (Same code as for RandomModel)
                        shape = tf.shape(input=actions[name])

                        # Action choices
                        choices = list(range(spec['num_values']))
                        choices_tile = ((1,) + spec['shape'] + (1,))
                        choices = np.tile(A=[choices], reps=choices_tile)
                        choices_shape = ((1,) + spec['shape'] + (spec['num_values'],))
                        choices = tf.constant(value=choices, dtype=int_dtype, shape=choices_shape)
                        ones = tf.ones(shape=(len(spec['shape']) + 1,), dtype=tf.int64)
                        batch_size = tf.dtypes.cast(x=shape[0:1], dtype=tf.int64)
                        multiples = tf.concat(values=(batch_size, ones), axis=0)
                        choices = tf.tile(input=choices, multiples=multiples)

                        # Random unmasked action
                        mask = auxiliaries[name + '_mask']
                        num_values = tf.math.count_nonzero(
                            input_tensor=mask, axis=-1, dtype=tf.int64
                        )
                        random_action = tf.random.uniform(shape=shape, dtype=float_dtype)
                        random_action = tf.dtypes.cast(
                            x=(random_action * tf.dtypes.cast(x=num_values, dtype=float_dtype)),
                            dtype=tf.int64
                        )

                        # Correct for masked actions
                        choices = tf.boolean_mask(tensor=choices, mask=mask)
                        offset = tf.math.cumsum(x=num_values, axis=-1, exclusive=True)
                        random_action = tf.gather(params=choices, indices=(random_action + offset))

                        # Random action
                        condition = tf.random.uniform(shape=shape, dtype=float_dtype) < exploration
                        return tf.where(condition=condition, x=random_action, y=actions[name])

                elif spec['type'] == 'float':
                    if 'min_value' in spec:

                        def apply_exploration():
                            noise = tf.random.normal(shape=shape, dtype=float_dtype) * exploration
                            return tf.clip_by_value(
                                t=(actions[name] + noise), clip_value_min=spec['min_value'],
                                clip_value_max=spec['max_value']
                            )

                    else:

                        def apply_exploration():
                            noise = tf.random.normal(shape=shape, dtype=float_dtype) * exploration
                            return actions[name] + noise

                skip_exploration = tf.math.logical_or(
                    x=deterministic, y=tf.math.equal(x=exploration, y=zero_float)
                )
                actions[name] = self.cond(
                    pred=skip_exploration, true_fn=no_exploration, false_fn=apply_exploration
                )
                dependencies = util.flatten(xs=actions)

        # Variable noise
        if len(variables) > 0:
            with tf.control_dependencies(control_inputs=dependencies):

                def reverse_variable_noise():
                    assignments = list()
                    for variable, noise in zip(variables, variable_noise_tensors):
                        assignments.append(variable.assign_sub(delta=noise, read_value=False))
                    with tf.control_dependencies(control_inputs=assignments):
                        return util.no_operation()

                reversed_variable_noise = self.cond(
                    pred=skip_variable_noise, true_fn=util.no_operation,
                    false_fn=reverse_variable_noise
                )
                dependencies = (reversed_variable_noise,)

        # Update states/internals/actions buffers
        with tf.control_dependencies(control_inputs=dependencies):

            def update_buffers():
                operations = list()
                buffer_index = self.buffer_index[parallel]
                for name in self.states_spec:
                    operations.append(
                        self.states_buffer[name].scatter_nd_update(
                            indices=[(parallel, buffer_index)], updates=states[name]
                        )
                    )
                for name in self.internals_spec:
                    operations.append(
                        self.internals_buffer[name].scatter_nd_update(
                            indices=[(parallel, buffer_index + one)], updates=internals[name]
                        )
                    )
                for name in self.auxiliaries_spec:
                    operations.append(
                        self.auxiliaries_buffer[name].scatter_nd_update(
                            indices=[(parallel, buffer_index)], updates=auxiliaries[name]
                        )
                    )
                for name in self.actions_spec:
                    operations.append(
                        self.actions_buffer[name].scatter_nd_update(
                            indices=[(parallel, buffer_index)], updates=actions[name]
                        )
                    )

                # Increment buffer index
                with tf.control_dependencies(control_inputs=operations):
                    incremented_buffer_index = self.buffer_index.scatter_nd_add(
                        indices=[(parallel,)], updates=[one]
                    )

                with tf.control_dependencies(control_inputs=(incremented_buffer_index,)):
                    return util.no_operation()

            updated_buffers = self.cond(
                pred=independent, true_fn=util.no_operation, false_fn=update_buffers
            )

        # Return timestep
        with tf.control_dependencies(control_inputs=(updated_buffers,)):
            # Function-level identity operation for retrieval (plus enforce dependency)
            for name, spec in self.actions_spec.items():
                actions[name] = util.identity_operation(
                    x=actions[name], operation_name=(name + '-output')
                )
            timestep = util.identity_operation(
                x=self.global_timestep, operation_name='timestep-output'
            )

        return actions, timestep

    def api_observe(self):
        # Inputs
        terminal = self.terminal_input
        reward = self.reward_input
        parallel = self.parallel_input

        zero = tf.constant(value=0, dtype=util.tf_dtype(dtype='long'))

        # Assertions
        assertions = [
            # terminal: type and shape
            tf.debugging.assert_type(tensor=terminal, tf_type=util.tf_dtype(dtype='long')),
            tf.debugging.assert_rank(x=terminal, rank=1),
            # reward: type and shape
            tf.debugging.assert_type(tensor=reward, tf_type=util.tf_dtype(dtype='float')),
            tf.debugging.assert_rank(x=reward, rank=1),
            # parallel: type, shape and value
            tf.debugging.assert_type(tensor=parallel, tf_type=util.tf_dtype(dtype='long')),
            tf.debugging.assert_scalar(tensor=parallel),
            tf.debugging.assert_non_negative(x=parallel),
            tf.debugging.assert_less(
                x=parallel,
                y=tf.constant(value=self.parallel_interactions, dtype=util.tf_dtype(dtype='long'))
            ),
            # shape of terminal equals shape of reward
            tf.debugging.assert_equal(x=tf.shape(input=terminal), y=tf.shape(input=reward)),
            # size of terminal equals buffer index
            tf.debugging.assert_equal(
                x=tf.shape(input=terminal, out_type=tf.int64)[0],
                y=tf.dtypes.cast(x=self.buffer_index[parallel], dtype=tf.int64)
            ),
            # at most one terminal
            tf.debugging.assert_less_equal(
                x=tf.math.count_nonzero(input_tensor=terminal, dtype=util.tf_dtype(dtype='long')),
                y=tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
            ),
            # if terminal, last timestep in batch
            tf.debugging.assert_equal(
                x=tf.math.reduce_any(input_tensor=tf.math.greater(x=terminal, y=zero)),
                y=tf.math.greater(x=terminal[-1], y=zero)
            )
        ]

        # Set global tensors
        Module.update_tensors(
            deterministic=tf.constant(value=True, dtype=util.tf_dtype(dtype='bool')),
            independent=tf.constant(value=False, dtype=util.tf_dtype(dtype='bool')),
            optimization=tf.constant(value=False, dtype=util.tf_dtype(dtype='bool')),
            timestep=self.global_timestep, episode=self.global_episode, update=self.global_update
        )

        with tf.control_dependencies(control_inputs=assertions):
            reward = self.add_summary(
                label=('timestep-reward', 'rewards'), name='timestep-reward', tensor=reward
            )
            assignment = self.episode_reward.scatter_nd_add(
                indices=[(parallel,)], updates=[tf.math.reduce_sum(input_tensor=reward, axis=0)]
            )

        # Increment episode
        def increment_episode():
            assignments = list()
            one = tf.constant(value=1, dtype=util.tf_dtype(dtype='long'))
            assignments.append(self.episode.scatter_nd_add(indices=[(parallel,)], updates=[one]))
            assignments.append(self.global_episode.assign_add(delta=one, read_value=False))
            zero_float = tf.constant(value=0.0, dtype=util.tf_dtype(dtype='float'))
            zero_float = self.add_summary(
                label=('episode-reward', 'rewards'), name='episode-reward',
                tensor=self.episode_reward[parallel], pass_tensors=zero_float, step='episode'
            )
            assignments.append(
                self.episode_reward.scatter_nd_update(indices=[(parallel,)], updates=[zero_float])
            )
            with tf.control_dependencies(control_inputs=assignments):
                return util.no_operation()

        with tf.control_dependencies(control_inputs=(assignment,)):
            incremented_episode = self.cond(
                pred=(terminal[-1] > zero), true_fn=increment_episode, false_fn=util.no_operation
            )
            dependencies = (incremented_episode,)

        # Preprocessing reward
        if 'reward' in self.preprocessing:
            with tf.control_dependencies(control_inputs=dependencies):
                reward = self.preprocessing['reward'].apply(x=reward)
            dependencies = (reward,)

        # Core observe: retrieve observe operation
        with tf.control_dependencies(control_inputs=dependencies):
            buffer_index = self.buffer_index[parallel]
            states = OrderedDict()
            for name in self.states_spec:
                states[name] = self.states_buffer[name][parallel, :buffer_index]
            internals = OrderedDict()
            for name in self.internals_spec:
                internals[name] = self.internals_buffer[name][parallel, :buffer_index]
            auxiliaries = OrderedDict()
            for name in self.auxiliaries_spec:
                auxiliaries[name] = self.auxiliaries_buffer[name][parallel, :buffer_index]
            actions = OrderedDict()
            for name in self.actions_spec:
                actions[name] = self.actions_buffer[name][parallel, :buffer_index]

            reward = self.add_summary(
                label=('raw-reward', 'rewards'), name='raw-reward', tensor=reward
            )

            is_updated = self.core_observe(
                states=states, internals=internals, auxiliaries=auxiliaries, actions=actions,
                terminal=terminal, reward=reward
            )

        # Reset buffer index
        with tf.control_dependencies(control_inputs=(is_updated,)):
            reset_buffer_index = self.buffer_index.scatter_nd_update(
                indices=[(parallel,)], updates=[zero]
            )
            dependencies = (reset_buffer_index,)

        if len(self.preprocessing) > 0:
            with tf.control_dependencies(control_inputs=dependencies):

                def reset_preprocessors():
                    operations = list()
                    for preprocessor in self.preprocessing.values():
                        operations.append(preprocessor.reset())
                    return tf.group(*operations)

                preprocessors_reset = self.cond(
                    pred=(terminal[-1] > zero), true_fn=reset_preprocessors,
                    false_fn=util.no_operation
                )
                dependencies = (preprocessors_reset,)

        # Return episode
        with tf.control_dependencies(control_inputs=dependencies):
            # Function-level identity operation for retrieval (plus enforce dependency)
            updated = util.identity_operation(x=is_updated, operation_name='updated-output')
            episode = util.identity_operation(
                x=self.global_episode, operation_name='episode-output'
            )
            update = util.identity_operation(
                x=self.global_update, operation_name='update-output'
            )

        return updated, episode, update

    def tf_core_act(self, states, internals, auxiliaries):
        raise NotImplementedError

    def tf_core_observe(self, states, internals, auxiliaries, actions, terminal, reward):
        raise NotImplementedError

    def tf_regularize(self, states, internals, auxiliaries):
        return super().tf_regularize()

    def save(self, directory=None, filename=None, append_timestep=True):
        if self.summarizer_spec is not None:
            self.monitored_session.run(fetches=self.summarizer_flush)

        if directory is None:
            assert self.saver_directory is not None
            directory = self.saver_directory
        if filename is None:
            filename = self.saver_filename
        save_path = os.path.join(directory, filename)

        return self.saver.save(
            sess=self.session, save_path=save_path,
            global_step=(self.global_timestep if append_timestep else None),
            # latest_filename=None,  # Defaults to 'checkpoint'.
            meta_graph_suffix='meta', write_meta_graph=True, write_state=True
        )

    def restore(self, directory=None, filename=None):
        if directory is None:
            assert self.saver_directory
            directory = self.saver_directory
        if filename is None or not os.path.isfile(os.path.join(directory, filename + '.meta')):
            save_path = tf.train.latest_checkpoint(checkpoint_dir=directory, latest_filename=None)
        else:
            save_path = os.path.join(directory, filename)

        self.saver.restore(sess=self.session, save_path=save_path)
        return self.reset()
