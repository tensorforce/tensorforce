*TensorForce - modular deep reinforcement learning in TensorFlow*
=================================================================================

.. image:: https://badges.gitter.im/reinforceio/TensorForce.svg
   :alt: Join the chat at https://gitter.im/reinforceio/TensorForce
   :target: https://gitter.im/reinforceio/TensorForce?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

Introduction
------------

TensorForce is an open source reinforcement learning library focused on
providing clear APIs, readability and modularisation to deploy
reinforcement learning solutions both in research and practice.
TensorForce is built on top of TensorFlow.

The main difference to existing libraries is a strict
separation of environments, agents and update logic that facilitates
usage in non-simulation environments. Further, research code often relies on fixed network
architectures that have been used to tackle particular benchmarks.
TensorForce is built with the idea that (almost) everything should be
optionally configurable and in particular uses value function template
configurations to be able to quickly experiment with new models. The
goal of TensorForce is to provide a practitioner's reinforcement
learning framework that integrates into modern software service
architectures.

TensorForce is actively being maintained and developed both to
continuously improve the existing code as well as to reflect new
developments as they arise (see road map for more). The aim is not to
include every new trick but to adopt methods as
they prove themselves stable, e.g. as of early 2017 hybrid A3C and TRPO
variants provide the basis for a lot of research. We also offer TensorForce
support through our Gitter channel.

Acknowledgements
----------------

The goal of TensorForce is not just to re-implement existing algorithms, but
to provide clear APIs and modularisations, and later provide serving,
integration and deployment components. The credit for original open
source implementations, which we have adopted and modified into our
architecture, fully belongs to the original authors, which have all made
their code available under MIT licenses.

In particular, credit goes to John Schulman, Ilya Sutskever and Wojciech
Zaremba for their various TRPO implementations, Rocky Duan for rllab,
Taehoon Kim for his DQN and NAF implementations, and many others who
have put in effort to make deep reinforcement learning more accessible
through blog posts and tutorials.

Features
--------

TensorForce currently integrates with the OpenAI Gym API, OpenAI
Universe and DeepMind lab. The following algorithms are available (all policy methods both continuous/discrete):

1. A3C using distributed TF
2. Trust Region Policy Optimization (TRPO) with generalised
   advantage estimation (GAE)
3. Normalised Advantage functions (NAFs)
4. DQN/Double-DQN
5. Vanilla Policy Gradients (VPG)

Installation
------------

For the most straight-forward install via pip, execute:

::

    git clone git@github.com:reinforceio/tensorforce.git
    cd tensorforce
    pip install -e .

To update TensorForce, just run ``git pull`` in the tensorforce
directory. Please not that we did not include OpenAI Gym/Universe/DeepMind lab in the default
install script because not everyone will want to use these. Please install them as required,
usually via pip.


Docker coming soon.

Examples and documentation
--------------------------

For a quick start, you can run one of our example scripts using the
provided configurations, e.g. to run the TRPO agent on CartPole, execute
from the examples folder:

::

    python tensorforce/examples/openai_gym.py CartPole-v0 -a TRPOAgent -c tensorforce/examples/configs/trpo_cartpole.json
    -n tensorforce/examples/configs/trpo_network_example.json
    
Documentation is available at `ReadTheDocs <http://tensorforce.readthedocs.io>`__.

Create and use agents
---------------------

To use TensorForce as a library without using the pre-defined simulation runners, simply install and import the library,
then create an agent and use it as seen below (see documentation for all optional parameters):

::

   from tensorforce.config import Config
   from tensorforce.util.agent_util import create_agent

   config = Config()

   # Set basic problem parameters
   config.batch_size = 1000
   config.state_shape = [10]
   config.actions = 5
   config.continuous = False

   # Define 2 fully connected layers
   config.network_layers = [{"type": "dense", "num_outputs": 50},
                            {"type": "dense", "num_outputs": 50}]

   # Create a Trust Region Policy Optimization agent
   agent = create_agent('TRPOAgent', config)

   # Get new data from somewhere, e.g. a client to a web app
   client = MyClient('http://127.0.0.1', 8080)

   # Poll new state from client
   input = client.get_state()

   # Get prediction from agent
   action = agent.get_action(input)

   # Do something with action
   reward = client.execute(action)

   # Add experience, agent automatically updates model according to batch size
   agent.add_observation(input, action, reward)



Update notes
------------

Coming soon/under construction:

- More examples scripts and predefined configurations for typical environments
- Example runner script for DeepMind lab
- Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic - `paper <https://arxiv.org/abs/1611.02247>`__


12th April 2017:

- Optional LSTM usage in network configurations, layer types "lstm", and type "flatten" to explicitly reshape between layers
  Support still experimental
- Cleaner internal state management for policy gradient models, including recurrent states via a list of
  internal states managed by the models.
- More consistent tensor shapes between models
- Log-level as config parameter, more informative logging for level INFO

Road map and contributions
--------------------------

TensorForce is still in alpha and hence continuously being updated.
Contributions are always welcome! We will use github issues to track
development. We ask that contributions integrate within the general code
style and architecture. For larger features it might be sensible to join
our Gitter chat or drop us an email to coordinate development. There is a very long list of
features, algorithms and infrastructure that we want to add over time and
we will prioritise this depending on our own research, community requests and contributions. The
larger road-map of things we would like to have (in no particular order) looks as follows:

1. More generic distributed/multi-threaded API
2. Hybrid A3C/policy gradient algorithms - not clear yet which
   combination method will work best, but a number of papers showcasing
   different approaches have been accepted to ICLR 2017.
3. A multi/sub-task API. An important topic in current research is to decompose larger tasks into
   a hierarchy of subtasks/auxiliary goals. Implementing new approaches in an easily configurable way for end-users
   will not be trivial and it might us take some time to get to it.
4. Transfer learning architectures (e.g. progressive neural networks, pathnet, ..).
5. RL serving components. TensorFlow serving can serve trained models but is not suitable to manage RL lifecycles.

Support and contact
-------------------

TensorForce is maintained by reinforce.io, a new project focused on
providing open source reinforcement learning infrastructure. For any
questions or support, get in touch at contact@reinforce.io.

You are also welcome to join our Gitter channel for help with using
TensorForce, bugs or contributions: `https://gitter.im/reinforceio/TensorForce <https://gitter.im/reinforceio/TensorForce>`__
