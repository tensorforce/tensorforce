Agent and model overview
========================

A reinforcement learning agent provides methods to process states and return actions, to store past observations, and to load and save models. Most agents employ a ``Model`` which implements the algorithms to calculate the next action given the current state and to update model parameters from past experiences.

  Environment <-> Runner <-> Agent <-> Model

Parameters to the agent are passed as a ``Config`` object or a ``dict``. The configuration is usually passed on to the ``Model``.

Ready-to-use agents and models
==============================

We implemented some of the most common RL algorithms and try to keep these up-to-date. Here we provide an overview over all implemented agents and models.

RLAgent / General parameters
----------------------------

``RLAgent`` is the base class for all reinforcement learning agents.
Every agent inherits from this class.



Parameters
~~~~~~~~~~

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
-----------

The ``MemoryAgent`` class implements a replay memory, from which it
samples batches to update the value function.

Parameters:
~~~~~~~~~~~

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
--------

Standard DQN agent (Minh et al., 2015) and DDQN agent. Uses a replay memory (inherits from ``MemoryAgent``).

Parameters:
~~~~~~~~~~~

Uses all `general parameters <#RLAgent>`__ and all parameters from the
`MemoryAgent <#MemoryAgent>`__. Additional parameters:

.. code:: python

    # Parameters for DQNModel
    config += {
        "double_dqn": bool,  # use double dqn network (or not)
        "tau": float,  # indicate how fast the target network should be updated
        "gamma": float,  # discount factor
        "alpha": float,  # learning rate
        "clip_gradients": bool,  # clip gradients
        "clip_value": float,  # clip values exceeding [-clip_value; clip_value]
    }

Building your own agent
=======================

If you want to build your own agent, it should always inherit from
``RLAgent``. If your agent uses a replay memory, it should probably
inherit from ``MemoryAgent``.

Reinforcement learning agents often differ
only by their respective value function. Extending the
MemoryAgent ist straightforward:

.. code:: python

    from tensorforce.rl_agents import MemoryAgent

    class MyAgent(MemoryAgent):
        name = 'MyAgent'

        model_ref = MyModel

``model_ref`` points to the model class. A model should always inherit from ``tensorforce.models.Model``.

.. code:: python

    import numpy as np
    import tensorforce as tf
    from tensorforce.models import Model
    from tensorforce.models.neural_networks import NeuralNetwork
    
    class MyModel(Model):
        default_config = {
            "test": 0.3
        }
    
        def __init__(self, config, scope):
            super(MyModel, self).__init__(config, scope)
            self.action_count = self.config.actions
            
            with tf.device(self.config.tf_device):
                # Create state placeholder
                self.state = tf.placeholder(tf.float32, self.batch_shape + list(self.config.state_shape), name="state")
                # Create neural network
                self.network = NeuralNetwork(self.config.network_layers, self.state, scope=self.scope + "network")
                self.network_out = self.network.get_output()
                # Create training operations
                self.create_ops()
                # Create optimizer
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.alpha)
            
        def get_action(self, state, episode=1):
            return np.random.randint(0, self.action_count)
            
        def update(self, batch):
            
            
        def create_ops(self):
            with tf.name_scope(self.scope):
                with tf.name_scope("predict"):
                    self.q_action = tf.argmax(self.nn_output, dimension
                    
                with tf.name_scope("update"):
                    loss = tf.reduce_sum(tf.square( - self.q_action))
                    self.update_op = self.optimizer.minimize(loss)
