TensorForce: A TensorFlow library for applied reinforcement learning
====================================================================

[![Docs](https://readthedocs.org/projects/tensorforce/badge)](http://tensorforce.readthedocs.io/en/latest/)
[![Gitter](https://badges.gitter.im/reinforceio/TensorForce.svg)](https://gitter.im/reinforceio/TensorForce?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/reinforceio/tensorforce.svg?branch=master)](https://travis-ci.org/reinforceio/tensorforce)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/reinforceio/tensorforce/blob/master/LICENSE)

Introduction
------------

TensorForce is an open source reinforcement learning library focused on
providing clear APIs, readability and modularisation to deploy
reinforcement learning solutions both in research and practice.
TensorForce is built on top of TensorFlow and compatible with Python 2.7
and &gt;3.5 and supports multiple state inputs and multi-dimensional
actions to be compatible with Gym, Universe, and DeepMind lab.

An introductory blog post can also be found [on our blog.](https://reinforce.io/blog/introduction-to-tensorforce)

IMPORTANT: Please do read the latest update notes at the bottom of this
document to get an idea of how the project is evolving, especially
concerning majorAPI breaking updates.

The main difference to existing libraries is a strict separation of
environments, agents and update logic that facilitates usage in
non-simulation environments. Further, research code often relies on
fixed network architectures that have been used to tackle particular
benchmarks. TensorForce is built with the idea that (almost) everything
should be optionally configurable and in particular uses value function
template configurations to be able to quickly experiment with new
models. The goal of TensorForce is to provide a practitioner's
reinforcement learning framework that integrates into modern software
service architectures.

TensorForce is actively being maintained and developed both to
continuously improve the existing code as well as to reflect new
developments as they arise (see road map for more). The aim is not to
include every new trick but to adopt methods as
they prove themselves stable.

Features
--------

TensorForce currently integrates with the OpenAI Gym API, OpenAI
Universe and DeepMind lab. The following algorithms are available (all
policy methods both continuous/discrete):

1.  A3C using distributed TensorFlow - now as part of our generic Model
    usable with different agents
2.  Trust Region Policy Optimization (TRPO) with generalised advantage
    estimation (GAE)
3.  Normalised Advantage functions (NAFs)
4.  DQN/Double-DQN
5.  Vanilla Policy Gradients (VPG)
6.  Deep Q-learning from Demonstration (DQFD) -
    [paper](https://arxiv.org/abs/1704.03732)

Installation
------------

For the most straight-forward install via pip, execute:

```bash
git clone git@github.com:reinforceio/tensorforce.git
cd tensorforce
pip install -e .
```

To update TensorForce, just run `git pull` in the tensorforce directory.
Please note that we did not include OpenAI Gym/Universe/DeepMind lab in
the default install script because not everyone will want to use these.
Please install them as required, usually via pip.

Examples and documentation
--------------------------

For a quick start, you can run one of our example scripts using the
provided configurations, e.g. to run the TRPO agent on CartPole, execute
from the examples folder:

```bash
python examples/openai_gym.py CartPole-v0 -a TRPOAgent -c examples/configs/trpo_cartpole.json -n examples/configs/trpo_cartpole_network.json
```

Documentation is available at
[ReadTheDocs](http://tensorforce.readthedocs.io). We also have tests
validating models on minimal environments which can be run from the main
directory by executing `pytest`{.sourceCode}.

Use with DeepMind lab
---------------------

Since DeepMind lab is only available as source code, a manual install
via bazel is required. Further, due to the way bazel handles external
dependencies, cloning TensorForce into lab is the most convenient way to
run it using the bazel BUILD file we provide. To use lab, first download
and install it according to instructions
<https://github.com/deepmind/lab/blob/master/docs/build.md>:

```bash
git clone https://github.com/deepmind/lab.git
```

Add to the lab main BUILD file:

```
package(default_visibility = ["//visibility:public"])
```

Clone TensorForce into the lab directory, then run the TensorForce bazel runner. Note that using any specific configuration file
currently requires changing the Tensorforce BUILD file to adjust environment parameters.

```bash
bazel run //tensorforce:lab_runner
```

Please note that we have not tried to reproduce any lab results yet, and
these instructions just explain connectivity in case someone wants to
get started there.

Create and use agents
---------------------

To use TensorForce as a library without using the pre-defined simulation
runners, simply install and import the library, then create an agent and
use it as seen below (see documentation for all optional parameters):

```python
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
state = client.get_state()

# Get prediction from agent, execute
action = agent.act(state=state)
reward = client.execute(action)

# Add experience, agent automatically updates model according to batch size
agent.observe(reward=reward, terminal=False)
```

Update notes
------------

8th July 2017

- BREAKING CHANGE: We modified the act and observe API once more because we think there was
a lack of clarity with regard to which state is observed (current vs next). The agent now internally
manages states and actions in the correct sequence so observe only needs reward and terminal.
- We further introduced a method ```import_observations``` so memory-based agents can preload
data into memory (e.g. if historic data is available). We also added a method ```last_observation```
on the generic agent which gives the current state, action, reward, terminal and internal state
- Fixed distributed agent mode, should run as intended now
- Fixed target network usage in NAF. Tests now run smoothl
- DQFDAgent now inherits from MemoryAgent


2nd July 2017

- Fixed lab integration: updated bazel BUILD file with command line options
- Adjusted environment integration to correctly select state and action interfaces
- Changed default agent to VPG since lab mixes continuous and discrete actions


25h June 2017

-   Added prioritised experience replay
-   Added RandomAgent for discrete/continuous random baselines
-   Moved pre-processing from runner to agent, analogue to exploration

11th June 2017

-   Fixed bug in DQFD test where demo data was not always the
    correct action. Also fixed small bug in DQFD loss (mean over
    supervised loss)
-   Network entry added to configuration so no separate network builder
    has to be passed to the agent constructor (see example)
-   The async mode using distributed tensorflow has been merged into the
    main model class. See the openai\_gym\_async.py example. In
    particular, this means multiple agents are now available in
    async mode. N.b. we are still working on making async/distributed
    things more convenient to use.
-   Fixed bug in NAF where target value (V) was connected to
    training output. Also added gradient clipping to NAF because we
    observed occasional numerical instability in testing.
-   For the same reason, we have altered the tests to always run
    multiple times and allow for an occasional failure on travis so our
    builds don't get broken by a random initialisation leading to
    an under/overflow.
-   Updated OpenAI Universe integration to work with our state/action
    interface, see an example in examples/openai\_universe.py
-   Added convenience method to create Network directly from json
    without needing to create a network builder, see examples for usage


Support and contact
-------------------

TensorForce is maintained by [reinforce.io](https://reinforce.io), a new
project focused on providing reinforcement learning software
infrastructure. For any questions or support, get in touch at
<contact@reinforce.io>.

You are also welcome to join our Gitter channel for help with using
TensorForce, bugs or contributions:
[<https://gitter.im/reinforceio/TensorForce>](https://gitter.im/reinforceio/TensorForce)


Cite
----

If you use TensorForce in your academic research, we would be grateful if you could cite it as follows:

```
@misc{schaarschmidt2017tensorforce,
    author = {Schaarschmidt, Michael and Kuhnle, Alexander and Fricke, Kai},
    title = {TensorForce: A TensorFlow library for applied reinforcement learning},
    howpublished={Web page},
    url = {https://github.com/reinforceio/tensorforce},
    year = {2017}
}
```
