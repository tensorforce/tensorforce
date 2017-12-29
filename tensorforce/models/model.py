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
It implements the `reset`, `act` and `update` functions, which give the interface the `Agent` class
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
* `action_exploration(action, exploration, action_spec)` for action postprocessing (e.g. exploration), returning the processed batch of actions.
* `preprocess_reward(states, internals, terminal, reward)` for reward preprocessing (e.g. reward normalization), returning the processed batch of rewards.
* `create_output_operations(states, internals, actions, terminal, reward, deterministic)` for further output operations, similar to the two above for `Model.act` and `Model.update`.
* `tf_optimization(states, internals, actions, terminal, reward)` for further optimization operations (e.g. the baseline update in a `PGModel` or the target network update in a `QModel`), returning a single grouped optimization operation.
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
from tensorforce.core.optimizers import Optimizer, GlobalOptimizer
from tensorforce.core.preprocessing import PreprocessorStack


class Model(object):
    """
    Base class for all (TensorFlow-based) models.
    """

    def __init__(
        self,
        states_spec,
        actions_spec,
        device,
        session_config,
        scope,
        saver_spec,
        summary_spec,
        distributed_spec,
        optimizer,
        discount,
        variable_noise,
        states_preprocessing_spec,
        explorations_spec,
        reward_preprocessing_spec
    ):
        # States and actions specifications
        self.states_spec = states_spec
        self.actions_spec = actions_spec

        # TensorFlow device and scope
        self.device = device
        self.session_config = session_config
        self.scope = scope

        # Saver/summary/distributed specifications
        self.saver_spec = saver_spec
        self.summary_spec = summary_spec
        self.distributed_spec = distributed_spec

        # TensorFlow summaries
        if summary_spec is None:
            self.summary_labels = set()
        else:
            self.summary_labels = set(summary_spec.get('labels', ()))

        # Optimizer
        self.optimizer = optimizer

        # Discount factor
        self.discount = discount

        # Variable noise
        assert variable_noise is None or variable_noise > 0.0
        self.variable_noise = variable_noise

        # Preprocessing and exploration
        self.states_preprocessing_spec = states_preprocessing_spec
        self.explorations_spec = explorations_spec
        self.reward_preprocessing_spec = reward_preprocessing_spec

        # Setup TensorFlow graph and session
        self.setup()

    def setup(self):
        """
        Sets up the TensorFlow model graph and initializes the TensorFlow session.
        """
        default_graph = None
        if self.distributed_spec is None:
            self.global_model = None
            self.graph = tf.Graph()
            default_graph = self.graph.as_default()
            default_graph.__enter__()

        elif self.distributed_spec.get('parameter_server'):
            if self.distributed_spec.get('replica_model'):
                raise TensorForceError("Invalid config value for distributed mode.")
            self.global_model = None
            self.graph = tf.Graph()
            default_graph = self.graph.as_default()
            default_graph.__enter__()

        elif self.distributed_spec.get('replica_model'):
            self.device = tf.train.replica_device_setter(
                worker_device=self.device,
                cluster=self.distributed_spec['cluster_spec']
            )
            self.global_model = None
            # Replica model is part of its parent model's graph, hence no new graph here.
            self.graph = tf.get_default_graph()

        else:
            graph = tf.Graph()
            default_graph = graph.as_default()
            default_graph.__enter__()
            # Global model.
            self.global_model = deepcopy(self)
            self.global_model.distributed_spec['replica_model'] = True
            self.global_model.setup()
            self.graph = graph

        with tf.device(device_name_or_function=self.device):

            # Variables and summaries
            self.variables = dict()
            self.all_variables = dict()
            self.registered_variables = set()
            self.summaries = list()

            def custom_getter(getter, name, registered=False, second=False, **kwargs):
                if registered:
                    self.registered_variables.add(name)
                elif name in self.registered_variables:
                    registered = True
                variable = getter(name=name, **kwargs)  # Top-level, hence no 'registered'
                if not registered:
                    self.all_variables[name] = variable
                    if kwargs.get('trainable', True) and not name.startswith('optimization'):
                        self.variables[name] = variable
                        if 'variables' in self.summary_labels:
                            summary = tf.summary.histogram(name=name, values=variable)
                            self.summaries.append(summary)
                return variable

            # Episode
            collection = self.graph.get_collection(name='episode')
            if len(collection) == 0:
                self.episode = tf.Variable(
                    name='episode',
                    dtype=util.tf_dtype('int'),
                    trainable=False,
                    initial_value=0
                )
                self.graph.add_to_collection(name='episode', value=self.episode)
            else:
                assert len(collection) == 1
                self.episode = collection[0]

            # Timestep
            collection = self.graph.get_collection(name='timestep')
            if len(collection) == 0:
                self.timestep = tf.Variable(
                    name='timestep',
                    dtype=util.tf_dtype('int'),
                    trainable=False,
                    initial_value=0
                )
                self.graph.add_to_collection(name='timestep', value=self.timestep)
                self.graph.add_to_collection(name=tf.GraphKeys.GLOBAL_STEP, value=self.timestep)
            else:
                assert len(collection) == 1
                self.timestep = collection[0]

            # Create placeholders, tf functions, internals, etc
            self.initialize(custom_getter=custom_getter)

            # Input tensors
            states = {name: tf.identity(input=state) for name, state in self.states_input.items()}
            states = self.fn_preprocess_states(states=states)
            states = {name: tf.stop_gradient(input=state) for name, state in states.items()}
            internals = [tf.identity(input=internal) for internal in self.internals_input]
            actions = {name: tf.identity(input=action) for name, action in self.actions_input.items()}
            terminal = tf.identity(input=self.terminal_input)
            reward = tf.identity(input=self.reward_input)
            reward = self.fn_preprocess_reward(states=states, internals=internals, terminal=terminal, reward=reward)
            reward = tf.stop_gradient(input=reward)

            # Optimizer
            kwargs_opt = dict()
            kwargs_opt['summaries']         =self.summaries
            kwargs_opt['summary_labels']    =self.summary_labels            
            if self.optimizer is None:
                pass
            elif self.distributed_spec is not None and \
                    not self.distributed_spec.get('parameter_server') and \
                    not self.distributed_spec.get('replica_model'):
                # If not internal global model
                self.optimizer = GlobalOptimizer(optimizer=self.optimizer)
            else:
                self.optimizer = Optimizer.from_spec(spec=self.optimizer, kwargs=kwargs_opt)

            # Create output fetch operations
            self.create_output_operations(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward,
                update=self.update_input,
                deterministic=self.deterministic_input
            )

            for name, action in self.actions_output.items():
                if name in self.explorations:
                    self.actions_output[name] = tf.cond(
                        pred=self.deterministic_input,
                        true_fn=(lambda: action),
                        false_fn=(lambda: self.fn_action_exploration(
                            action=action,
                            exploration=self.explorations[name],
                            action_spec=self.actions_spec[name]
                        ))
                    )

            if any(k in self.summary_labels for k in ['inputs','states','actions','rewards']): 
                if any(k in self.summary_labels for k in ['inputs','states']):               
                    for name, state in states.items():
                        summary = tf.summary.histogram(name=(self.scope + '/inputs/states/' + name), values=state)
                        self.summaries.append(summary)
                if any(k in self.summary_labels for k in ['inputs','actions']):                                       
                    for name, action in actions.items():
                        summary = tf.summary.histogram(name=(self.scope + '/inputs/actions/' + name), values=action)
                        self.summaries.append(summary)
                if any(k in self.summary_labels for k in ['inputs','rewards']):                                       
                    summary = tf.summary.histogram(name=(self.scope + '/inputs/rewards'), values=reward)
                    self.summaries.append(summary)

        if self.distributed_spec is not None:
            if self.distributed_spec.get('replica_model'):
                # If internal global model
                return

            elif self.distributed_spec.get('parameter_server'):
                server = tf.train.Server(
                    server_or_cluster_def=self.distributed_spec['cluster_spec'],
                    job_name='ps',
                    task_index=self.distributed_spec['task_index'],
                    protocol=self.distributed_spec.get('protocol'),
                    config=None,
                    start=True
                )
                # Param server does nothing actively
                server.join()
                return

        # Global and local variables initialize operations
        if self.distributed_spec is None:
            global_variables = self.get_variables(include_non_trainable=True) + [self.episode, self.timestep]
            init_op = tf.variables_initializer(var_list=global_variables)
            ready_op = tf.report_uninitialized_variables(var_list=global_variables)
            ready_for_local_init_op = None
            local_init_op = None
        else:
            global_variables = self.global_model.get_variables(include_non_trainable=True) + [self.episode, self.timestep]
            local_variables = self.get_variables(include_non_trainable=True) + [self.episode, self.timestep]
            init_op = tf.variables_initializer(var_list=global_variables)
            ready_op = tf.report_uninitialized_variables(var_list=(global_variables + local_variables))
            ready_for_local_init_op = tf.report_uninitialized_variables(var_list=global_variables)
            local_init_op = tf.group(*(local_var.assign(value=global_var) for local_var, global_var in zip(local_variables, global_variables)))

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

        # TensorFlow saver object
        saver = tf.train.Saver(
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
            saver=saver,
            copy_from_scaffold=None
        )

        hooks = list()

        # Checkpoint saver hook
        if self.saver_spec is not None and (self.distributed_spec is None or self.distributed_spec['task_index'] == 0):
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
        if self.summary_spec is None:
            self.summary_writer_hook = None
        else:
            # TensorFlow summary writer object
            self.summary_writer = tf.summary.FileWriter(
                logdir=self.summary_spec['directory'],
                graph=self.graph,
                max_queue=10,
                flush_secs=120,
                filename_suffix=None
            )
            self.summary_writer_hook = util.UpdateSummarySaverHook(
                update_input=self.update_input,
                save_steps=self.summary_spec.get('steps'),  # Either one or the other has to be set.
                save_secs=self.summary_spec.get('seconds', None if 'steps' in self.summary_spec else 120),
                output_dir=None,  # None since given via 'summary_writer' argument.
                summary_writer=self.summary_writer,
                scaffold=self.scaffold,
                summary_op=None  # None since given via 'scaffold' argument.
            )
            hooks.append(self.summary_writer_hook)

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

        if self.distributed_spec is None:
            # TensorFlow non-distributed monitored session object
            self.monitored_session = tf.train.SingularMonitoredSession(
                hooks=hooks,
                scaffold=self.scaffold,
                master='',  # Default value.
                config=self.session_config,  # always the same?
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

            if self.distributed_spec['task_index'] == 0:
                # TensorFlow chief session creator object
                session_creator = tf.train.ChiefSessionCreator(
                    scaffold=self.scaffold,
                    master=server.target,
                    config=self.session_config,
                    checkpoint_dir=None,
                    checkpoint_filename_with_path=None
                )
            else:
                # TensorFlow worker session creator object
                session_creator = tf.train.WorkerSessionCreator(
                    scaffold=self.scaffold,
                    master=server.target,
                    config=self.session_config,
                )

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

        # # tf.ConfigProto(device_filters=['/job:ps', '/job:worker/task:{}/cpu:0'.format(self.task_index)])
        #         # config=tf.ConfigProto(device_filters=["/job:ps"])
        #         # config=tf.ConfigProto(
        #         #     inter_op_parallelism_threads=2,
        #         #     log_device_placement=True
        #         # )

    def close(self):
        if self.saver_directory is not None:
            self.save(append_timestep=True)
        self.monitored_session.close()

    def initialize(self, custom_getter):
        """
        Creates the TensorFlow placeholders and functions for this model. Moreover adds the  
        internal state placeholders and initialization values to the model.

        Args:
            custom_getter: The `custom_getter_` object to use for `tf.make_template` when creating TensorFlow functions.
        """

        # States preprocessing
        self.states_preprocessing = dict()
        if self.states_preprocessing_spec is None:
            for name, state in self.states_spec.items():
                state['processed_shape'] = state['shape']
        elif isinstance(self.states_preprocessing_spec, list):
            for name, state in self.states_spec.items():
                preprocessing = PreprocessorStack.from_spec(spec=self.states_preprocessing_spec)
                self.states_preprocessing[name] = preprocessing
                state['processed_shape'] = preprocessing.processed_shape(shape=state['shape'])
        else:
            for name, state in self.states_spec.items():
                if self.states_preprocessing_spec.get(name) is not None:
                    preprocessing = PreprocessorStack.from_spec(spec=self.states_preprocessing_spec[name])
                    self.states_preprocessing[name] = preprocessing
                    state['processed_shape'] = preprocessing.processed_shape(shape=state['shape'])
                else:
                    state['processed_shape'] = state['shape']

        # States
        self.states_input = dict()
        for name, state in self.states_spec.items():
            self.states_input[name] = tf.placeholder(
                dtype=util.tf_dtype(state['type']),
                shape=(None,) + tuple(state['shape']),
                name=name
            )

        # Actions
        self.actions_input = dict()
        for name, action in self.actions_spec.items():
            self.actions_input[name] = tf.placeholder(
                dtype=util.tf_dtype(action['type']),
                shape=(None,) + tuple(action['shape']),
                name=name
            )

        # Explorations
        self.explorations = dict()
        if self.explorations_spec is None:
            pass
        elif isinstance(self.explorations_spec, list):
            for name, state in self.actions_spec.items():
                self.explorations[name] = Exploration.from_spec(spec=self.explorations_spec)
        # single spec for all components of our action space
        elif "type" in self.explorations_spec:
            for name, state in self.actions_spec.items():
                self.explorations[name] = Exploration.from_spec(spec=self.explorations_spec)
        # different spec for different components of our action space
        else:
            for name, state in self.actions_spec.items():
                if self.explorations_spec.get(name) is not None:
                    self.explorations[name] = Exploration.from_spec(spec=self.explorations_spec[name])

        # Terminal
        self.terminal_input = tf.placeholder(dtype=util.tf_dtype('bool'), shape=(None,), name='terminal')

        # Reward preprocessing
        if self.reward_preprocessing_spec is None:
            self.reward_preprocessing = None
        else:
            self.reward_preprocessing = PreprocessorStack.from_spec(spec=self.reward_preprocessing_spec)
            if self.reward_preprocessing.processed_shape(shape=()) != ():
                raise TensorForceError("Invalid reward preprocessing!")

        # Reward
        self.reward_input = tf.placeholder(dtype=util.tf_dtype('float'), shape=(None,), name='reward')

        # Internal states
        self.internals_input = list()
        self.internals_init = list()

        # Deterministic action flag
        self.deterministic_input = tf.placeholder(dtype=util.tf_dtype('bool'), shape=(), name='deterministic')

        # Update flag
        self.update_input = tf.placeholder(dtype=util.tf_dtype('bool'), shape=(), name='update')

        # TensorFlow functions
        self.fn_discounted_cumulative_reward = tf.make_template(
            name_=(self.scope + '/discounted-cumulative-reward'),
            func_=self.tf_discounted_cumulative_reward,
            custom_getter_=custom_getter
        )
        self.fn_actions_and_internals = tf.make_template(
            name_=(self.scope + '/actions-and-internals'),
            func_=self.tf_actions_and_internals,
            custom_getter_=custom_getter
        )
        self.fn_loss_per_instance = tf.make_template(
            name_=(self.scope + '/loss-per-instance'),
            func_=self.tf_loss_per_instance,
            custom_getter_=custom_getter
        )
        self.fn_regularization_losses = tf.make_template(
            name_=(self.scope + '/regularization-losses'),
            func_=self.tf_regularization_losses,
            custom_getter_=custom_getter
        )
        self.fn_loss = tf.make_template(
            name_=(self.scope + '/loss'),
            func_=self.tf_loss,
            custom_getter_=custom_getter
        )
        self.fn_optimization = tf.make_template(
            name_=(self.scope + '/optimization'),
            func_=self.tf_optimization,
            custom_getter_=custom_getter
        )
        self.fn_preprocess_states = tf.make_template(
            name_=(self.scope + '/preprocess-states'),
            func_=self.tf_preprocess_states,
            custom_getter_=custom_getter
        )
        self.fn_action_exploration = tf.make_template(
            name_=(self.scope + '/action-exploration'),
            func_=self.tf_action_exploration,
            custom_getter_=custom_getter
        )
        self.fn_preprocess_reward = tf.make_template(
            name_=(self.scope + '/preprocess-reward'),
            func_=self.tf_preprocess_reward,
            custom_getter_=custom_getter
        )

        self.summary_configuration_op = None
        if self.summary_spec and 'meta_param_recorder_class' in self.summary_spec:
            self.summary_configuration_op = self.summary_spec['meta_param_recorder_class'].build_metagraph_list()
          
        # self.fn_summarization = tf.make_template(
        #     name_='summarization',
        #     func_=self.tf_summarization,
        #     custom_getter_=custom_getter
        # )

    def tf_preprocess_states(self, states):
        """
        Applies optional pre-processing to the states.
        """
        for name, state in states.items():
            if name in self.states_preprocessing:
                states[name] = self.states_preprocessing[name].process(tensor=state)
            else:
                states[name] = tf.identity(input=state)

        return states

    def tf_action_exploration(self, action, exploration, action_spec):
        """
        Applies optional exploration to the action.
        """
        action_shape = tf.shape(input=action)
        exploration_value = exploration.tf_explore(
            episode=self.episode,
            timestep=self.timestep,
            action_shape=action_shape
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
            action += tf.reshape(tensor=exploration_value,
                                 shape=tuple(1 for _ in range(action_shape.get_shape().as_list()[0])))
            if 'min_value' in action_spec:
                action = tf.clip_by_value(
                    t=action,
                    clip_value_min=action_spec['min_value'],
                    clip_value_max=action_spec['max_value']
                )

        return action

    def tf_preprocess_reward(self, states, internals, terminal, reward):
        """
        Applies optional pre-processing to the reward.
        """
        if self.reward_preprocessing is None:
            reward = tf.identity(input=reward)
        else:
            reward = self.reward_preprocessing.process(tensor=reward)

        return reward

    def tf_discounted_cumulative_reward(self, terminal, reward, discount, final_reward=0.0):
        """
        Creates the TensorFlow operations for calculating the discounted cumulative rewards
        for a given sequence of rewards.

        Args:
            terminal: Terminal boolean tensor.
            reward: Reward tensor.
            discount: Discount factor.
            final_reward: Last reward value in the sequence.

        Returns:
            Discounted cumulative reward tensor.
        """

        # TODO: n-step cumulative reward (particularly for envs without terminal)

        def cumulate(cumulative, reward_and_terminal):
            rew, term = reward_and_terminal
            return tf.where(
                condition=term,
                x=rew,
                y=(rew + cumulative * discount)
            )

        # Reverse since reward cumulation is calculated right-to-left, but tf.scan only works left-to-right
        reward = tf.reverse(tensor=reward, axis=(0,))
        terminal = tf.reverse(tensor=terminal, axis=(0,))

        reward = tf.scan(fn=cumulate, elems=(reward, terminal), initializer=final_reward)

        return tf.reverse(tensor=reward, axis=(0,))

    def tf_actions_and_internals(self, states, internals, update, deterministic):
        """
        Creates the TensorFlow operations for retrieving the actions (and posterior internal states)
        in reaction to the given input states (and prior internal states).

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            update: Boolean tensor indicating whether this call happens during an update.
            deterministic: Boolean tensor indicating whether action should be chosen  
                deterministically.

        Returns:
            Actions and list of posterior internal state tensors.
        """
        raise NotImplementedError

    def tf_loss_per_instance(self, states, internals, actions, terminal, reward, update):
        """
        Creates the TensorFlow operations for calculating the loss per batch instance
        of the given input states and actions.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.
            update: Boolean tensor indicating whether this call happens during an update.

        Returns:
            Loss tensor.
        """
        raise NotImplementedError

    def tf_regularization_losses(self, states, internals, update):
        """
        Creates the TensorFlow operations for calculating the regularization losses for the given input states.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            update: Boolean tensor indicating whether this call happens during an update.

        Returns:
            Dict of regularization loss tensors.
        """
        return dict()

    def tf_loss(self, states, internals, actions, terminal, reward, update):
        # Mean loss per instance
        loss_per_instance = self.fn_loss_per_instance(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward,
            update=update
        )
        loss = tf.reduce_mean(input_tensor=loss_per_instance, axis=0)

        # Loss without regularization summary
        if 'losses' in self.summary_labels:
            summary = tf.summary.scalar(name='loss-without-regularization', tensor=loss)
            self.summaries.append(summary)

        # Regularization losses
        losses = self.fn_regularization_losses(states=states, internals=internals, update=update)
        if len(losses) > 0:
            loss += tf.add_n(inputs=list(losses.values()))
            if 'regularization' in self.summary_labels:
                for name, loss_val in losses.items():
                    summary = tf.summary.scalar(name="regularization/"+name, tensor=loss_val)
                    self.summaries.append(summary)

        # Total loss summary
        if 'losses' in self.summary_labels or 'total-loss' in self.summary_labels:
            summary = tf.summary.scalar(name='total-loss', tensor=loss)
            self.summaries.append(summary)

        return loss

    def get_optimizer_kwargs(self, states, internals, actions, terminal, reward, update):
        """
        Returns the optimizer arguments including the time, the list of variables to optimize,
        and various argument-free functions (in particular `fn_loss` returning the combined
        0-dim batch loss tensor) which the optimizer might require to perform an update step.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.
            update: Boolean tensor indicating whether this call happens during an update.

        Returns:
            Loss tensor of the size of the batch.
        """
        kwargs = dict()
        kwargs['time'] = self.timestep
        kwargs['variables'] = self.get_variables()
        kwargs['fn_loss'] = (
            lambda: self.fn_loss(states=states, internals=internals, actions=actions, terminal=terminal, reward=reward, update=update)
        )
        if self.global_model is not None:
            kwargs['global_variables'] = self.global_model.get_variables()
        return kwargs

    def tf_optimization(self, states, internals, actions, terminal, reward, update):
        """
        Creates the TensorFlow operations for performing an optimization update step based
        on the given input states and actions batch.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.
            update: Boolean tensor indicating whether this call happens during an update.

        Returns:
            The optimization operation.
        """
        if self.optimizer is None:
            return tf.no_op()
        else:
            optimizer_kwargs = self.get_optimizer_kwargs(
                states=states,
                internals=internals,
                actions=actions,
                terminal=terminal,
                reward=reward,
                update=update
            )
            return self.optimizer.minimize(**optimizer_kwargs)

    def create_output_operations(self, states, internals, actions, terminal, reward, update, deterministic):
        """
        Calls all the relevant TensorFlow functions for this model and hence creates all the
        TensorFlow operations involved.

        Args:
            states: Dict of state tensors.
            internals: List of prior internal state tensors.
            actions: Dict of action tensors.
            terminal: Terminal boolean tensor.
            reward: Reward tensor.
            update: Boolean tensor indicating whether this call happens during an update.
            deterministic: Boolean tensor indicating whether action should be chosen  
                deterministically.
        """

        # Create graph by calling the functions corresponding to model.act() / model.update(), to initialize variables.
        # TODO: Could call reset here, but would have to move other methods below reset.
        self.fn_actions_and_internals(
            states=states,
            internals=internals,
            update=update,
            deterministic=deterministic
        )
        self.fn_loss_per_instance(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward,
            update=update
        )

        # Tensor fetched for model.act()
        operations = list()
        if self.variable_noise is not None and self.variable_noise > 0.0:
            # Add variable noise
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
                update=update,
                deterministic=deterministic
            )

        # Increment timestep
        increment_timestep = tf.shape(input=next(iter(states.values())))[0]
        increment_timestep = self.timestep.assign_add(delta=increment_timestep)
        operations = [increment_timestep]

        # Subtract variable noise
        if self.variable_noise is not None and self.variable_noise > 0.0:
            for variable, noise_delta in zip(self.get_variables(), noise_deltas):
                operations.append(variable.assign_sub(delta=noise_delta))

        with tf.control_dependencies(control_inputs=operations):
            # Trivial operation to enforce control dependency
            self.timestep_output = self.timestep + 0

        # Tensor fetched for model.observe()
        increment_episode = tf.count_nonzero(input_tensor=terminal, dtype=util.tf_dtype('int'))
        increment_episode = self.episode.assign_add(delta=increment_episode)
        with tf.control_dependencies(control_inputs=(increment_episode,)):
            self.increment_episode = self.episode + 0
        # TODO: add up rewards per episode and add summary_label 'episode-reward'

        # Tensor(s) fetched for model.update()
        self.optimization = self.fn_optimization(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward,
            update=update
        )
        self.loss_per_instance = self.fn_loss_per_instance(
            states=states,
            internals=internals,
            actions=actions,
            terminal=terminal,
            reward=reward,
            update=update
        )

    def get_variables(self, include_non_trainable=False):
        """
        Returns the TensorFlow variables used by the model.

        Returns:
            List of variables.
        """

        if include_non_trainable:
                # optimizer variables and timestep/episode only included if 'include_non_trainable' set
            model_variables = [self.all_variables[key] for key in sorted(self.all_variables)]
            states_preprocessing_variables = [
                variable for name in self.states_preprocessing.keys()
                for variable in self.states_preprocessing[name].get_variables()
            ]
            explorations_variables = [
                variable for name in self.explorations.keys()
                for variable in self.explorations[name].get_variables()
            ]
            if self.reward_preprocessing is not None:
                reward_preprocessing_variables = self.reward_preprocessing.get_variables()
            else:
                reward_preprocessing_variables = list()
            if self.optimizer is None:
                optimizer_variables = list()
            else:
                optimizer_variables = self.optimizer.get_variables()

            return model_variables + states_preprocessing_variables + explorations_variables + \
                reward_preprocessing_variables + optimizer_variables

        else:
            return [self.variables[key] for key in sorted(self.variables)]

    def get_summaries(self):
        """
        Returns the TensorFlow summaries reported by the model

        Returns:
            List of summaries
        """
        return self.summaries

    def reset(self):
        """
        Resets the model to its initial state on episode start.

        Returns:
            Current episode and timestep counter, and a list containing the internal states  
            initializations.
        """
        #TODO preprocessing reset call moved from agent
        episode, timestep = self.monitored_session.run(fetches=(self.episode, self.timestep))
        return episode, timestep, list(self.internals_init)

    def act(self, states, internals, deterministic=False):
        fetches = [self.actions_output, self.internals_output, self.timestep_output]

        name = next(iter(self.states_spec))
        batched = (np.asarray(states[name]).ndim != len(self.states_spec[name]['shape']))
        if batched:
            feed_dict = {state_input: states[name] for name, state_input in self.states_input.items()}
            feed_dict.update({internal_input: internals[n] for n, internal_input in enumerate(self.internals_input)})
        else:
            feed_dict = {state_input: (states[name],) for name, state_input in self.states_input.items()}
            feed_dict.update({internal_input: (internals[n],) for n, internal_input in enumerate(self.internals_input)})

        feed_dict[self.deterministic_input] = deterministic
        feed_dict[self.update_input] = False

        actions, internals, timestep = self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)

        if not batched:
            actions = {name: action[0] for name, action in actions.items()}
            internals = [internal[0] for internal in internals]

        if self.summary_configuration_op is not None:
            summary_values = self.session.run(self.summary_configuration_op)
            self.summary_writer.add_summary(summary_values)
            self.summary_writer.flush()
            # Only do this operation once to reduce duplicate data in Tensorboard
            self.summary_configuration_op = None

        return actions, internals, timestep

    def observe(self, terminal, reward):
        fetches = self.increment_episode

        terminal = np.asarray(terminal)
        batched = (terminal.ndim == 1)
        if batched:
            feed_dict = {self.terminal_input: terminal, self.reward_input: reward, }
        else:
            feed_dict = {self.terminal_input: (terminal,), self.reward_input: (reward,)}

        feed_dict[self.update_input] = False

        episode = self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)

        return episode

    def update(self, states, internals, actions, terminal, reward, return_loss_per_instance=False):
        fetches = [self.optimization]

        # Optionally fetch loss per instance
        if return_loss_per_instance:
            fetches.append(self.loss_per_instance)

        terminal = np.asarray(terminal)
        batched = (terminal.ndim == 1)
        if batched:
            feed_dict = {state_input: states[name] for name, state_input in self.states_input.items()}
            feed_dict.update(
                {internal_input: internals[n]
                    for n, internal_input in enumerate(self.internals_input)}
            )
            feed_dict.update(
                {action_input: actions[name]
                    for name, action_input in self.actions_input.items()}
            )
            feed_dict[self.terminal_input] = terminal
            feed_dict[self.reward_input] = reward
        else:
            feed_dict = {state_input: (states[name],) for name, state_input in self.states_input.items()}
            feed_dict.update(
                {internal_input: (internals[n],)
                    for n, internal_input in enumerate(self.internals_input)}
            )
            feed_dict.update(
                {action_input: (actions[name],)
                    for name, action_input in self.actions_input.items()}
            )
            feed_dict[self.terminal_input] = (terminal,)
            feed_dict[self.reward_input] = (reward,)

        feed_dict[self.deterministic_input] = True
        feed_dict[self.update_input] = True

        fetched = self.monitored_session.run(fetches=fetches, feed_dict=feed_dict)

        if return_loss_per_instance:
            return fetched[1]

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
            Checkpoint path were the model was saved.
        """
        if self.summary_writer_hook is not None:
            self.summary_writer_hook._summary_writer.flush()

        return self.scaffold.saver.save(
            sess=self.session,
            save_path=(self.saver_directory if directory is None else directory),
            global_step=(self.timestep if append_timestep else None),
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

        self.scaffold.saver.restore(sess=self.session, save_path=file)
