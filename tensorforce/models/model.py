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


"""
The `Model` class coordinates the creation and execution of all TensorFlow operations within a model.
It implements the `reset`, `act` and `update` functions, which form the interface the `Agent` class
communicates with, and which should not need to be overwritten. Instead, the following TensorFlow
functions need to be implemented:

* `tf_actions_and_internals(states, internals, deterministic)` returning the batch of
   actions and successor internal states.
* `tf_loss_per_instance(states, internals, actions, terminal, reward)` returning the loss
   per instance for a batch.

Further, the following TensorFlow functions should be extended accordingly:

* `initialize(custom_getter)` defining TensorFlow placeholders/functions and adding internal states.
* `get_variables()` returning the list of TensorFlow variables (to be optimized) of this model.
* `tf_regularization_losses(states, internals)` returning a dict of regularization losses.
* `get_optimizer_kwargs(states, internals, actions, terminal, reward)` returning a dict of potential
   arguments (argument-free functions) to the optimizer.

Finally, the following TensorFlow functions can be useful in some cases:

* `preprocess_states(states)` for state preprocessing, returning the processed batch of states.
* `tf_action_exploration(action, exploration, action_spec)` for action postprocessing (e.g. exploration),
    returning the processed batch of actions.
* `tf_preprocess_reward(states, internals, terminal, reward)` for reward preprocessing (e.g. reward normalization),
    returning the processed batch of rewards.
* `create_output_operations(states, internals, actions, terminal, reward, deterministic)` for further output operations,
    similar to the two above for `Model.act` and `Model.update`.
* `tf_optimization(states, internals, actions, terminal, reward)` for further optimization operations
    (e.g. the baseline update in a `PGModel` or the target network update in a `QModel`),
    returning a single grouped optimization operation.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from copy import deepcopy
import os

import numpy as np
import tensorflow as tf

from tensorforce import TensorForceError, util
from tensorforce.core.explorations import Exploration
from tensorforce.core.preprocessors import PreprocessorStack


class Model(object):
    """
    Base class for all (TensorFlow-based) models.
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
        reward_preprocessing
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
        """
        # Network crated from network_spec in distribution_model.py
        # Needed for named_tensor access
        self.network = None

        # States/internals/actions specifications
        self.states_spec = states
        self.internals_spec = dict()
        self.actions_spec = actions

        # TensorFlow scope, device
        self.scope = scope
        self.device = device

        # Saver/summaries/distributes
        if saver is None or saver.get('directory') is None:
            self.saver_spec = None
        else:
            self.saver_spec = saver
        if summarizer is None or summarizer.get('directory') is None:
            self.summarizer_spec = None
        else:
            self.summarizer_spec = summarizer

        self.execution_spec = execution
        # TODO (SVEN) Modify to read other execution spec args here
        if self.execution_spec:
            self.session_config = self.execution_spec.get('session_config', None)
            # Default single-process execution.
            self.execution_type = self.execution_spec.get('type', 'single')
            self.distributed_spec = self.execution_spec.get('distributed_spec', None)
        else:
            self.execution_type = 'single'
            self.session_config = None
            self.distributed_spec = None

        # TensorFlow summaries
        if self.summarizer_spec is None:
            self.summary_labels = set()
        else:
            self.summary_labels = set(self.summarizer_spec.get('labels', ()))

        # Batching capacity for act/observe interface
        assert batching_capacity is None or (isinstance(batching_capacity, int) and batching_capacity > 0)
        self.batching_capacity = batching_capacity

        # Variable noise
        assert variable_noise is None or variable_noise > 0.0
        self.variable_noise = variable_noise

        # Preprocessing and exploration
        self.states_preprocessing_spec = states_preprocessing
        self.actions_exploration_spec = actions_exploration
        self.reward_preprocessing_spec = reward_preprocessing

        self.is_observe = False

        self.states_preprocessing = None
        self.actions_exploration = None
        self.reward_preprocessing = None

        self.variables = None
        self.all_variables = None
        self.registered_variables = None
        self.summaries = None

        self.timestep = None
        self.episode = None
        self.global_timestep = None
        self.global_episode = None

        self.states_input = None
        self.internals_input = None
        self.actions_input = None
        self.terminal_input = None
        self.reward_input = None
        self.deterministic_input = None
        self.independent_input = None
        self.update_input = None
        self.internals_init = None

        self.fn_initialize = None
        self.fn_actions_and_internals = None
        self.fn_observe_timestep = None
        self.fn_action_exploration = None

        self.graph = None
        self.global_model = None
        self.saver = None
        self.scaffold = None
        self.saver_directory = None
        self.session = None
        self.monitored_session = None
        self.summary_writer = None
        self.summary_writer_hook = None

        self.increment_episode = None

        self.actions_output = None
        self.internals_output = None
        self.timestep_output = None

        self.summary_configuration_op = None

        # Setup TensorFlow graph and session
        self.setup()

    def setup(self):
        """
        Sets up the TensorFlow model graph and initializes (and enters) the TensorFlow session.
        """

        # Create our Graph or figure out, which shared/global one to use.
        default_graph = None
        # No parallel RL or ThreadedRunner with Hogwild! shared network updates:
        # Build single graph and work with that from here on. In the case of threaded RL, the central
        # and already initialized model is handed to the worker Agents via the ThreadedRunner's
        # WorkerAgentGenerator factory.
        if self.execution_type == 'single':
            self.graph = tf.Graph()
            default_graph = self.graph.as_default()
            default_graph.__enter__()
            self.global_model = None
        # Distributed tensorflow setup (each process gets its own (identical) graph).
        # We are the parameter server.
        elif self.execution_type == 'distributed' and self.distributed_spec.get('parameter_server'):
            if self.distributed_spec.get('replica_model'):
                raise TensorForceError("Invalid config value for distributed mode.")
            self.graph = tf.Graph()
            default_graph = self.graph.as_default()
            default_graph.__enter__()
            self.global_model = None
            self.scope = self.scope + '-ps'
        # We are a worker's replica model.
        # Place our ops round-robin on all worker devices.
        elif self.execution_type == 'distributed' and self.distributed_spec.get('replica_model'):
            self.graph = tf.get_default_graph()
            self.global_model = None
            # The graph is the parent model's graph, hence no new graph here.
            self.device = tf.train.replica_device_setter(
                worker_device=self.device,
                cluster=self.distributed_spec['cluster_spec']
            )
            self.scope = self.scope + '-ps'
        # We are a worker:
        # Construct the global model (deepcopy of ourselves), set it up via `setup` and link to it (global_model).
        else:
            graph = tf.Graph()
            default_graph = graph.as_default()
            default_graph.__enter__()
            self.global_model = deepcopy(self)
            self.global_model.distributed_spec['replica_model'] = True
            self.global_model.setup()
            self.graph = graph
            self.as_local_model()
            self.scope = self.scope + '-worker' + str(self.distributed_spec['task_index'])

        with tf.device(device_name_or_function=self.device):
            with tf.variable_scope(name_or_scope=self.scope, reuse=False):

                # Variables and summaries
                self.variables = dict()
                self.all_variables = dict()
                self.registered_variables = set()
                self.summaries = list()

                def custom_getter(getter, name, registered=False, **kwargs):
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

                # Global timestep
                collection = self.graph.get_collection(name='global-timestep')
                if len(collection) == 0:
                    self.global_timestep = tf.Variable(
                        name='global-timestep',
                        dtype=util.tf_dtype('int'),
                        trainable=False,
                        initial_value=0
                    )
                    self.graph.add_to_collection(name='global-timestep', value=self.global_timestep)
                    self.graph.add_to_collection(name=tf.GraphKeys.GLOBAL_STEP, value=self.global_timestep)
                else:
                    assert len(collection) == 1
                    self.global_timestep = collection[0]

                # Global episode
                collection = self.graph.get_collection(name='global-episode')
                if len(collection) == 0:
                    self.global_episode = tf.Variable(
                        name='global-episode',
                        dtype=util.tf_dtype('int'),
                        trainable=False,
                        initial_value=0
                    )
                    self.graph.add_to_collection(name='global-episode', value=self.global_episode)
                else:
                    assert len(collection) == 1
                    self.global_episode = collection[0]

                # Create placeholders, tf functions, internals, etc
                self.initialize(custom_getter=custom_getter)

                # self.fn_actions_and_internals(
                #     states=states,
                #     internals=internals,
                #     update=update,
                #     deterministic=deterministic
                # )
                # self.fn_loss_per_instance(
                #     states=states,
                #     internals=internals,
                #     actions=actions,
                #     terminal=terminal,
                #     reward=reward,
                #     update=update
                # )
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

        if self.execution_type == 'single':
            global_variables = self.get_variables(include_submodules=True, include_nontrainable=True)
            global_variables += [self.global_episode, self.global_timestep]
            init_op = tf.variables_initializer(var_list=global_variables)
            ready_op = tf.report_uninitialized_variables(var_list=global_variables)
            ready_for_local_init_op = None
            local_init_op = None

        else:
            # We are just a replica model: Return.
            if self.distributed_spec.get('replica_model'):
                return
            # We are the parameter server: Start and wait.
            elif self.distributed_spec.get('parameter_server'):
                server = tf.train.Server(
                    server_or_cluster_def=self.distributed_spec['cluster_spec'],
                    job_name='ps',
                    task_index=self.distributed_spec['task_index'],
                    protocol=self.distributed_spec.get('protocol'),
                    config=None,
                    start=True
                )
                # Param server does nothing actively.
                server.join()
                return

            # Global and local variable initializers.
            global_variables = self.global_model.get_variables(
                include_submodules=True,
                include_nontrainable=True
            )
            global_variables += [self.global_episode, self.global_timestep]
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

        # Summary operation
        summaries = self.get_summaries()
        if len(summaries) > 0:
            summary_op = tf.summary.merge(inputs=summaries)
        else:
            summary_op = None

        for c in self.get_savable_components():
            c.register_saver_ops()

        # TensorFlow saver object
        self.saver = tf.train.Saver(
            var_list=global_variables,  # should be given?
            reshape=False,
            sharded=False,  # should be true?
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

        # TensorFlow scaffold object
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

        if self.execution_type == 'single':
            # TensorFlow non-distributed monitored session object
            self.monitored_session = tf.train.SingularMonitoredSession(
                hooks=hooks,
                scaffold=self.scaffold,
                master='',  # Default value.
                config=self.session_config,  # self.execution_spec.get('session_config'),
                checkpoint_dir=None
            )

        else:
            server = tf.train.Server(
                server_or_cluster_def=self.distributed_spec['cluster_spec'],
                job_name='worker',
                task_index=self.distributed_spec['task_index'],
                protocol=self.distributed_spec.get('protocol'),
                config=self.session_config,
                start=True
            )

            # if self.execution_spec['task_index'] == 0:
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

        if default_graph:
            default_graph.__exit__(None, None, None)
        self.graph.finalize()
        self.monitored_session.__enter__()
        self.session = self.monitored_session._tf_sess()

    def close(self):
        if self.saver_directory is not None:
            self.save(append_timestep=True)
        self.monitored_session.close()

    def as_local_model(self):
        pass

    def initialize(self, custom_getter):
        """
        Creates the TensorFlow placeholders and functions for this model. Moreover adds the
        internal state placeholders and initialization values to the model.

        Args:
            custom_getter: The `custom_getter_` object to use for `tf.make_template` when creating TensorFlow functions.
        """

        # States
        self.states_input = dict()
        for name, state in self.states_spec.items():
            self.states_input[name] = tf.placeholder(
                dtype=util.tf_dtype(state['type']),
                shape=(None,) + tuple(state['shape']),
                name=('state-' + name)
            )

        # States preprocessing
        self.states_preprocessing = dict()

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
        # single preprocessor for all components of our state space
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

        # Internals
        self.internals_input = dict()
        self.internals_init = dict()
        for name, internal in self.internals_spec.items():
            self.internals_input[name] = tf.placeholder(
                dtype=util.tf_dtype(internal['type']),
                shape=(None,) + tuple(internal['shape']),
                name=('internal-' + name)
            )
            if internal['initialization'] == 'zeros':
                self.internals_init[name] = np.zeros(shape=internal['shape'])
            else:
                raise TensorForceError("Invalid internal initialization value.")

        # Actions
        self.actions_input = dict()
        for name, action in self.actions_spec.items():
            self.actions_input[name] = tf.placeholder(
                dtype=util.tf_dtype(action['type']),
                shape=(None,) + tuple(action['shape']),
                name=('action-' + name)
            )

        # Actions exploration
        self.actions_exploration = dict()
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
        if self.reward_preprocessing_spec is None:
            self.reward_preprocessing = None
        else:
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

        # TensorFlow functions
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

        self.summary_configuration_op = None
        if self.summarizer_spec and 'meta_param_recorder_class' in self.summarizer_spec:
            self.summary_configuration_op = self.summarizer_spec['meta_param_recorder_class'].build_metagraph_list()

        # self.fn_summarization = tf.make_template(
        #     name_='summarization',
        #     func_=self.tf_summarization,
        #     custom_getter_=custom_getter
        # )

    def tf_initialize(self):
        # Timestep
        self.timestep = tf.get_variable(
            name='timestep',
            dtype=util.tf_dtype('int'),
            initializer=0,
            trainable=False
        )

        # Episode
        self.episode = tf.get_variable(
            name='episode',
            dtype=util.tf_dtype('int'),
            initializer=0,
            trainable=False
        )

        if self.batching_capacity is None:
            capacity = 1
        else:
            capacity = self.batching_capacity

        # States buffer variable
        self.states_buffer = dict()
        for name, state in self.states_spec.items():
            self.states_buffer[name] = tf.get_variable(
                name=('state-' + name),
                shape=((capacity,) + tuple(state['shape'])),
                dtype=util.tf_dtype(state['type']),
                trainable=False
            )

        # Internals buffer variable
        self.internals_buffer = dict()
        for name, internal in self.internals_spec.items():
            self.internals_buffer[name] = tf.get_variable(
                name=('internal-' + name),
                shape=((capacity,) + tuple(internal['shape'])),
                dtype=util.tf_dtype(internal['type']),
                trainable=False
            )

        # Actions buffer variable
        self.actions_buffer = dict()
        for name, action in self.actions_spec.items():
            self.actions_buffer[name] = tf.get_variable(
                name=('action-' + name),
                shape=((capacity,) + tuple(action['shape'])),
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
            internals: List of prior internal state tensors.
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
        Creates the TensorFlow operations for performing the observation of a full time step's
        information.

        Args:
            states (dict): Dict of state tensors (each key represents one state space component).
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.

        Returns:
            The observation operation.
        """
        raise NotImplementedError

    def create_act_operations(self, states, internals, deterministic, independent):
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
            return self.global_timestep

        # Normal act followed by observe, with additional operations.
        def normal_act():
            # Store current states, internals and actions
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
                return self.global_timestep + 0

        # Only increment timestep and update buffer if act not independent
        self.timestep_output = tf.cond(pred=independent, true_fn=independent_act, false_fn=normal_act)

    def create_observe_operations(self, terminal, reward):
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
            # Reset index
            reset_index = tf.assign(ref=self.buffer_index, value=0)

        with tf.control_dependencies(control_inputs=(reset_index,)):
            # Trivial operation to enforce control dependency
            self.episode_output = self.global_episode + 0

        # TODO: add up rewards per episode and add summary_label 'episode-reward'

    def create_operations(self, states, internals, actions, terminal, reward, deterministic, independent):
        """
        Creates output operations for acting, observing and interacting with the memory.
        """
        self.create_act_operations(
            states=states,
            internals=internals,
            deterministic=deterministic,
            independent=independent
        )
        self.create_observe_operations(reward=reward, terminal=terminal)

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
            assert self.batching_capacity is not None and state.shape[0] <= self.batching_capacity

        fetches = [self.actions_output, self.internals_output, self.timestep_output]
        if self.network is not None and fetch_tensors is not None:
            for name in fetch_tensors:
                valid, tensor = self.network.get_named_tensor(name)
                if valid:
                    fetches.append(tensor)
                else:
                    keys=self.network.get_list_of_named_tensor()
                    raise TensorForceError('Cannot fetch named tensor "{}", Available {}.'.format(name,keys))

        #     feed_dict = {state_input: states[name] for name, state_input in self.states_input.items()}
        #     feed_dict.update({internal_input: internals[n] for n, internal_input in enumerate(self.internals_input)})
        # else:
        #     feed_dict = {state_input: (states[name],) for name, state_input in self.states_input.items()}
        #     feed_dict.update({internal_input: (internals[n],) for n, internal_input in enumerate(self.internals_input)})

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
            terminal (bool): Whether the episode has terminated.
            reward (float): The observed reward value.

        Returns:
            The value of the model-internal episode counter.
        """
        # terminal = np.asarray(terminal)
        # batched = (terminal.ndim == 1)

        fetches = self.episode_output

        feed_dict = self.get_feed_dict(terminal=terminal, reward=reward)

        # if batched:
        #     assert self.batching_capacity is not None and terminal.shape[0] <= self.batching_capacity
        #     feed_dict = {self.terminal_input: terminal, self.reward_input: reward, }
        # else:
        #     feed_dict = {self.terminal_input: (terminal,), self.reward_input: (reward,)}

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
