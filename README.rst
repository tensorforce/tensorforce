*TensorForce - modular deep reinforcement learning in TensorFlow*
=================================================================

.. |logo1| image:: https://badges.gitter.im/reinforceio/TensorForce.svg
           :scale: 100%
           :target: https://gitter.im/reinforceio/TensorForce?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
           :align: top
.. |logo2| image:: https://travis-ci.org/reinforceio/tensorforce.svg?branch=master
   :scale: 50%
   :align: top

+---------+---------+
| |logo1| | |logo2| |
+---------+---------+


Introduction
------------

TensorForce is an open source reinforcement learning library focused on
providing clear APIs, readability and modularisation to deploy
reinforcement learning solutions both in research and practice.
TensorForce is built on top of TensorFlow and compatible with Python 2.7 and >3.5 and
supports multiple state inputs and multi-dimensional actions to be compatible with Gym, Universe,
and DeepMind lab.

IMPORTANT: Please do read the latest update notes at the bottom of this document to get an idea of
how the project is evolving, especially concerning majorAPI breaking updates.

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

Features
--------

TensorForce currently integrates with the OpenAI Gym API, OpenAI
Universe and DeepMind lab. The following algorithms are available (all policy methods both continuous/discrete):

1. A3C using distributed TensorFlow - now as part of our generic Model usable with different agents
2. Trust Region Policy Optimization (TRPO) with generalised
   advantage estimation (GAE)
3. Normalised Advantage functions (NAFs)
4. DQN/Double-DQN
5. Vanilla Policy Gradients (VPG)
6. Deep Q-learning from Demonstration (DQFD) - `paper <https://arxiv.org/abs/1704.03732>`__

Installation
------------

For the most straight-forward install via pip, execute:

::

    git clone git@github.com:reinforceio/tensorforce.git
    cd tensorforce
    pip install -e .

To update TensorForce, just run ``git pull`` in the tensorforce
directory. Please note that we did not include OpenAI Gym/Universe/DeepMind lab in the default
install script because not everyone will want to use these. Please install them as required,
usually via pip.


Docker coming soon.

Examples and documentation
--------------------------

For a quick start, you can run one of our example scripts using the
provided configurations, e.g. to run the TRPO agent on CartPole, execute
from the examples folder:

::

    python examples/openai_gym.py CartPole-v0 -a TRPOAgent -c examples/configs/trpo_cartpole.json -n examples/configs/trpo_cartpole_network.json

Documentation is available at `ReadTheDocs <http://tensorforce.readthedocs.io>`__. We also have tests validating models
on minimal environments which can be run from the main directory by executing :code:`pytest`.

Use with DeepMind lab
---------------------

Since DeepMind lab is only available as source code, a manual install via bazel is required. Further, due to the way bazel handles external dependencies,
cloning TensorForce into lab is the most convenient way to run it using the bazel BUILD file we provide. To use lab, first download and install it
according to instructions https://github.com/deepmind/lab/blob/master/docs/build.md:

::

   git clone https://github.com/deepmind/lab.git

Add to the lab main BUILD file:

::

   package(default_visibility = ["//visibility:public"])

Clone TensorForce into the lab directory, then run the TensorForce bazel runner. Note that using any specific configuration file requires
changing the Tensorforce BUILD file to tell bazel to include the new file in the build (just change the filenames in the data
line).

::

   bazel run //tensorforce:lab_runner


Please note that we have not tried to reproduce any lab results yet, and these instructions just explain connectivity
in case someone wants to get started there.


Create and use agents
---------------------

To use TensorForce as a library without using the pre-defined simulation runners, simply install and import the library,
then create an agent and use it as seen below (see documentation for all optional parameters):

::

  from tensorforce import Configuration
  from tensorforce.agents import TRPOAgent
  from tensorforce.core.networks import layered_network_builder

  config = Configuration(
    batch_size=100,
    state=dict(shape=(10,)),
    actions=dict(continuous=False, num_actions=2)
    network=layered_network_builder([dict(type='dense', size=50), dict(type='dense', size=50)])
  )

  # Create a Trust Region Policy Optimization agent
  agent = TRPOAgent(config=config)

  # Get new data from somewhere, e.g. a client to a web app
  client = MyClient('http://127.0.0.1', 8080)

  # Poll new state from client
  input = client.get_state()

  # Get prediction from agent, execute
  action = agent.act(input)
  reward = client.execute(action)

  # Add experience, agent automatically updates model according to batch size
  agent.observe(state=input, action=action, reward=reward, terminal=False)


Update notes
------------

11th June 2017

- Fixed bug in DQFD test where demo data was not always the correct action. Also fixed small bug in DQFD loss
  (mean over supervised loss)
- Network entry added to configuration so no separate network builder has to be passed to the agent constructor (see example)
- The async mode using distributed tensorflow has been merged into the main model class. See the openai_gym_async.py example.
  In particular, this means multiple agents are now available in async mode. N.b. we are still working on making async/distributed
  things more convenient to use.
- Fixed bug in NAF where target value (V) was connected to training output. Also added gradient clipping to NAF because we
  observed occasional numerical instability in testing.
- For the same reason, we have altered the tests to always run multiple times and allow for an occasional failure on travis so
  our builds don't get broken by a random initialisation leading to an under/overflow.
- Updated OpenAI Universe integration to work with our state/action interface, see an example in examples/openai_universe.py
- Added convenience method to create Network directly from json without needing to create a network builder, see examples for
  usage


29th May 2017

BREAKING CHANGES 0.2: We completely restructured the project to reduce redundant code, significantly improve execution time, allow
for multiple states and actions per step (by wrapping them in dicts), and much more. We are aware not everything is working
smoothly yet so please bear with us (or help by filing an issue). 0.1 still works. Following this rewrite, the  high level API should be stable going forward.
The most significant changes are listed below:

- RlAgent (now Agent) API change: add_observation() to observe(), get_action to act()
- Code reorganised to contain a folder "core" which contains common RL abstractions.
- States and actions are now conceptualised as dictionaries to support multiple state inputs and multiple actions of different shape
  per time step. In particular, this allows us to have a generic interface between gym, universe, lab and other potential environments
- External environments (tensorforce/external) have to implement the 'states' and 'actions' properties to define
  environment shapes.
- Models now all create their TensorFlow operations by calling the same function (create_tf_operations()). This will allow
  us to do useful things like wrapping these calls with TensorFlow device mappings.
- Minimal test environments are also implemented under external/environments for consistency. Please note
  that these tests are only meant to ensure the act and update mechanisms run in principle to help us make changes,
  they cannot replace running full environments
- Examples moved into separate directory
- N.b. we have not been able to test DeepMind lab yet
- The distributed_pg_model/distributed_agent have been deprecated. We want a general parallel/distributed functionality that
  works for as many models as possible, such as PAAC. We have started adding this functionality in to the main model
  but this is still work in progress.
- We will soon publish a blog post detailing the overall architecture and explaining some of our design
  choices



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

1. More generic distributed/multi-threaded API (e.g. PAAC)
2. Hybrid A3C/policy gradient algorithms - not clear yet which
   combination method will work best, but a number of papers showcasing
   different approaches have been accepted to ICLR 2017.
3. A multi/sub-task API. An important topic in current research is to decompose larger tasks into
   a hierarchy of subtasks/auxiliary goals. Implementing new approaches in an easily configurable way for end-users
   will not be trivial and it might us take some time to get to it.
4. Experimental Transfer learning architectures (e.g. progressive neural networks, pathnet, ..).
5. RL serving components. TensorFlow serving can serve trained models but is not suitable to manage RL lifecycles.

Support and contact
-------------------

TensorForce is maintained by `reinforce.io <https://reinforce.io>`__, a new project focused on
providing open source reinforcement learning infrastructure. For any
questions or support, get in touch at contact@reinforce.io.

You are also welcome to join our Gitter channel for help with using
TensorForce, bugs or contributions: `https://gitter.im/reinforceio/TensorForce <https://gitter.im/reinforceio/TensorForce>`__

Acknowledgements
----------------

The goal of TensorForce is not just to re-implement existing algorithms, but
to provide clear APIs and modularisations, and later provide serving,
integration and deployment components. Credit for some of the open
source implementations we have adopted and modified into our
architecture fully belongs to the original authors, which have all made
their code available under MIT licenses.

In particular, credit goes to John Schulman, Ilya Sutskever and Wojciech
Zaremba for their various TRPO implementations, Rocky Duan for rllab,
Taehoon Kim for his DQN and NAF implementations, and many others who
have put in effort to make deep reinforcement learning more accessible
through blog posts and tutorials.
