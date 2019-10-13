Tensorforce: a TensorFlow library for applied reinforcement learning
====================================================================

Tensorforce is an open-source deep reinforcement learning framework, with an emphasis on modularized flexible library design and straightforward usability for applications in research and practice. Tensorforce is built on top of `Google's TensorFlow framework <https://www.tensorflow.org/>`_ and compatible with Python 3 (Python 2 support was dropped with version 0.5).

Tensorforce follows a set of high-level design choices which differentiate it from other similar libraries:

- **Modular component-based design**: Feature implementations, above all, strive to be as generally applicable and configurable as possible, potentially at some cost of faithfully resembling details of the introducing paper.
- **Separation of RL algorithm and application**: Algorithms are agnostic to the type and structure of inputs (states/observations) and outputs (actions/decisions), as well as the interaction with the application environment.
- **Full-on TensorFlow models**: The entire reinforcement learning logic, including control flow, is implemented in TensorFlow, to enable portable computation graphs independent of application programming language, and to facilitate the deployment of models.



.. toctree::
  :maxdepth: 0
  :caption: Basics

  basics/installation
  basics/getting-started
  basics/specification
  basics/features
  basics/run
  basics/tune


.. toctree::
  :maxdepth: 0
  :caption: Agents

  agents/agent
  agents/constant
  agents/random
  agents/tensorforce
  agents/dqn
  agents/dueling_dqn
  agents/vpg
  agents/ac
  agents/a2c
  agents/dpg
  agents/ppo
  agents/trpo


.. toctree::
   :maxdepth: 1
   :caption: Modules

   modules/distributions
   modules/layers
   modules/memories
   modules/networks
   modules/objectives
   modules/optimizers
   modules/parameters
   modules/preprocessing
   modules/policies


.. toctree::
   :maxdepth: 0
   :caption: Environments

   environments/environment
   environments/ale
   environments/maze_explorer
   environments/open_sim
   environments/openai_gym
   environments/openai_retro
   environments/ple
   environments/vizdoom
