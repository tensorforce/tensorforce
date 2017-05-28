Agent and model overview
========================

A reinforcement learning agent provides methods to process states and return actions, to store past observations, and to load and save models. Most agents employ a ``Model`` which implements the algorithms to calculate the next action given the current state and to update model parameters from past experiences.

  Environment <-> Runner <-> Agent <-> Model

Parameters to the agent are passed as a ``Config`` object or a ``dict``. The configuration is usually passed on to the ``Model``.

Ready-to-use agents and models
------------------------------

We implemented some of the most common RL algorithms and try to keep these up-to-date. Here we provide an overview over all implemented agents and models.

Agent / General parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Agent`` is the base class for all reinforcement learning agents.
Every agent inherits from this class.



Parameters
""""""""""

.. code:: python

    from tensorforce.config import Config

    # General parameters
    config = Config({
        "tf_device": string  # tensorflow device
        "state_shape": tuple,  # shape of state tensor
        "action_shape": tuple,  # shape of action tensor
        "state_type": type,  # type of state tensor (default: np.float32)
        "action_type": type,  # type of action tensor (default: np.int)
        "reward_type": type,  # type of reward signal (default: np.float32)
        "deterministic_mode": bool  # Use a deterministic random function (default: False)
    })

MemoryAgent
~~~~~~~~~~~

The ``MemoryAgent`` class implements a replay memory, from which it
samples batches to update the value function.

Parameters:
"""""""""""

.. code:: python

    # Parameters for MemoryAgent
    config += {
        "memory_capacity": int,  # maxmium capacity for replay memory
        "batch_size": int,  # batch size for updates
        "update_rate": float,  # how often to update the value function (e.g. 0.25 means every 4th step)
        "use_target_network": bool,  # does the model use a target network?
        "target_network_update_rate": float,  # how often to update the target network
        "min_replay_size": int,  # minimum replay memory size to reach before first update
        "update_repeat": int  # how often to repeat updates
    }

DQNAgent
~~~~~~~~

Standard DQN agent (`Minh et al., 2015 <http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html>`__) and DDQN (`van Hasselt et al., 2015 <https://arxiv.org/abs/1509.06461>`__) agent. Uses a replay memory (inherits from ``MemoryAgent``).

Parameters:
"""""""""""

Uses all `general parameters <#Agent>`__ and all parameters from the
`MemoryAgent <#MemoryAgent>`__. Additional parameters:

.. code:: python

    # Parameters for DQNModel
    config += {
        "alpha": float,  # learning rate
        "gamma": float,  # discount factor
        "tau": float,  # indicates how fast the target network should be updated
        "double_dqn": bool,  # use double dqn network (or not)
        "clip_gradients": bool,  # clip gradients
        "clip_value": float,  # clip values exceeding [-clip_value; clip_value]
    }

NAFAgent
~~~~~~~~

Agent using Normalized Advantage Functions (NAF; `Gu et al., 2016 <https://arxiv.org/abs/1603.00748>`__) for continuous actions. Uses a replay memory (inherits from ``MemoryAgent``).

Parameters:
"""""""""""

Uses all `general parameters <#Agent>`__ and all parameters from the
`MemoryAgent <#MemoryAgent>`__. Additional parameters:

.. code:: python

    # Parameters for NAFModel
    config += {
        "alpha": float,  # learning rate
        "gamma": float,  # discount factor
        "tau": float,  # indicates how fast the target network should be updated
    }

PGAgent
~~~~~~~

Policy Gradient base agent and model. The agent collects experiences until conditions for an update are satisfied and then passes these to an updater. In particular,
the PGAgent internally manages the batching process so users do not have to.
`PGAgent` inherits from `Agent`.


Parameters:
"""""""""""

Uses all `general parameters <#Agent>`__. Additional parameters:

.. code:: python

    # Parameters for PGAgent
    config += {
        "batch_size": int,  # batch size for updates
    }
    
    # Parameters for PGModel
    config += {
        "alpha": float,  # learning rate
        "gamma": float,  # discount factor
        "use_gae": boolean,  # use general advantage estimation
        "gae_gamma": float,  # discount factor used in gae computation
        "normalize_advantage": boolean  # Normalize advantage
    }


VPGAgent
~~~~~~~~

Vanilla Policy Gradient agent and model. `VPGAgent` inherits from `PGAgent`.


Parameters:
"""""""""""

Uses all `general parameters <#Agent>`__ and all parameters from the
`PGAgent <#PGAgent>`__.

TRPOAgent
~~~~~~~~~

Trust Region Policy Optimization (`Schulman et al., 2015 <https://arxiv.org/abs/1502.05477>`__) agent and model. `TRPO` inherits from `PGAgent`.


Parameters:
"""""""""""

Uses all `general parameters <#Agent>`__ and all parameters from the
`PGAgent <#PGAgent>`__. Additional parameters:

.. code:: python
    
    # Parameters for TRPOModel
    config += {
        "cg_iterations": int,  # conjugate gradient interations
        "cg_damping": float,  # damping factor for the Fisher matrix
        "line_search_steps": int,  # line search steps
        "max_kl_divergence": float,  # maximum kl divergence
    }


Building your own agent
-----------------------

If you want to build your own agent, it should always inherit from
``Agent``. If your agent uses a replay memory, it should probably
inherit from ``MemoryAgent``.

Reinforcement learning agents often differ
only by their respective value function. Extending the
MemoryAgent ist straightforward:

.. code:: python

    # Full code at tensorforce/examples/simple_q_agent.py
    from tensorforce.agents import MemoryAgent

    class SimpleQAgent(MemoryAgent):
        """
        Simple agent extending MemoryAgent
        """
        name = 'SimpleQAgent'
    
        model_ref = SimpleQModel
    
        default_config = {
            "memory_capacity": 1000,  # hold the last 100 observations in the replay memory
            "batch_size": 10,  # train model with batches of 10
            "update_rate": 0.5,  # update parameters every other step
            "update_repeat": 1,  # repeat update only one time
            "min_replay_size": 0 # minimum size of replay memory before updating
        }

``model_ref`` points to the model class. A model should always inherit from ``tensorforce.models.Model``.

.. code:: python

    # Full code at tensorforce/examples/simple_q_agent.py
    import numpy as np
    import tensorforce as tf
    from tensorforce.models import Model
    from tensorforce.models.neural_networks import NeuralNetwork
    from tensorforce.config import Config
    
    class SimpleQModel(Model):
        # Default config values
        default_config = {
            "alpha": 0.01,
            "gamma": 0.99,
            "network_layers": [{
                "type": "linear",
                "num_outputs": 16
            }]
        }
    
        def __init__(self, config, scope):
            """
            Initialize model, build network and tensorflow ops
    
            :param config: Config object or dict
            :param scope: tensorflow scope name
            """
            super(SimpleQModel, self).__init__(config, scope)
            self.action_count = self.config.actions
    
            self.random = np.random.RandomState()
    
            with tf.device(self.config.tf_device):
                # Create state placeholder
                self.state = tf.placeholder(tf.float32, [None] + list(self.config.state_shape), name="state")
    
                # Create neural network
                output_layer = [{"type": "linear", "num_outputs": self.action_count}]
    
                define_network = NeuralNetwork.layered_network(self.config.network_layers + output_layer)
                self.network = NeuralNetwork(define_network, [self.state], scope=self.scope + 'network')
                self.network_out = self.network.output
    
                # Create operations
                self.create_ops()
                self.init_op = tf.global_variables_initializer()
    
                # Create optimizer
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.alpha)
    
                self.session.run(self.init_op)
    
        def get_action(self, state, episode=1):
            """
            Get action for a given state
    
            :param state: ndarray containing the state
            :param episode: number of episode (for epsilon decay and alike)
            :return: action
            """
    
            # self.exploration is initialized in Model.__init__ and provides an API for different explorations methods,
            # such as epsilon greedy.
            epsilon = self.exploration(episode, self.total_states)  # returns a float
    
            if self.random.random_sample() < epsilon:
                action = self.random.randint(0, self.action_count)
            else:
                action = self.session.run(self.q_action, {
                    self.state: [state]
                })[0]
    
            self.total_states += 1
            return action
    
        def update(self, batch):
            """
            Update model parameters
    
            :param batch: replay_memory batch
            :return:
            """
            # Get Q values for next states
            next_q = self.session.run(self.network_out, {
                self.state: batch['next_states']
            })
    
            # Bellmann equation Q = r + y * Q'
            q_targets = batch['rewards'] + (1. - batch['terminals'].astype(float)) \
                                           * self.config.gamma * np.max(next_q, axis=1)
    
            self.session.run(self.optimize_op, {
                self.state: batch['states'],
                self.actions: batch['actions'],
                self.q_targets: q_targets
            })
    
        def initialize(self):
            """
            Initialize model variables
            :return:
            """
            self.session.run(self.init_op)
    
        def create_ops(self):
            """
            Create tensorflow ops
    
            :return:
            """
            with tf.name_scope(self.scope):
                with tf.name_scope("predict"):
                    self.q_action = tf.argmax(self.network_out, axis=1)
    
                with tf.name_scope("update"):
                    # These are the target Q values, i.e. the actual rewards plus the expected values of the next states
                    # (Bellman equation).
                    self.q_targets = tf.placeholder(tf.float32, [None], name='q_targets')
    
                    # Actions that have been taken.
                    self.actions = tf.placeholder(tf.int32, [None], name='actions')
    
                    # We need the Q values of the current states to calculate the difference ("loss") between the
                    # expected values and the new values (q targets). Therefore we do a forward-pass
                    # and reduce the results to the actions that have been taken.
    
                    # One_hot tensor of the actions that have been taken.
                    actions_one_hot = tf.one_hot(self.actions, self.action_count, 1.0, 0.0, name='action_one_hot')
    
                    # Training output, reduced to the actions that have been taken.
                    q_values_actions_taken = tf.reduce_sum(self.network_out * actions_one_hot, axis=1,
                                                           name='q_acted')
    
                    # The loss is the difference between the q_targets and the expected q values.
                    self.loss = tf.reduce_sum(tf.square(self.q_targets - q_values_actions_taken))
                    self.optimize_op = self.optimizer.minimize(self.loss)
