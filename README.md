TensorForce: A TensorFlow library for applied reinforcement learning
====================================================================

[![Docs](https://readthedocs.org/projects/tensorforce/badge)](http://tensorforce.readthedocs.io/en/latest/)
[![Gitter](https://badges.gitter.im/reinforceio/TensorForce.svg)](https://docs.google.com/forms/d/1_UD5Pb5LaPVUviD0pO0fFcEnx_vwenvuc00jmP2rRIc/)
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

More information on architecture can also be found [on our blog.](https://reinforce.io/blog/)

Please do read the latest update notes (UPDATE_NOTES.md) for an idea of how the project is evolving, especially
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
developments as they arise. The aim is not to
include every new trick but to adopt methods as
they prove themselves stable.

Features
--------

TensorForce currently integrates with the OpenAI Gym API, OpenAI
Universe, DeepMind lab, ALE and Maze explorer. The following algorithms are available (all
policy methods both continuous/discrete and using a Beta distribution for bounded actions). 

-  A3C using distributed TensorFlow or a multithreaded runner - now as part of our generic Model
    usable with different agents. - [paper](https://arxiv.org/pdf/1602.01783.pdf)
- Trust Region Policy Optimization (TRPO) - ```trpo_agent``` - [paper](https://arxiv.org/abs/1502.05477)
- Normalised Advantage functions (NAFs) - ```naf_agent``` - [paper](https://arxiv.org/pdf/1603.00748.pdf)
- DQN - [paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)
- Double-DQN - ```ddqn_agent``` - [paper](https://arxiv.org/abs/1509.06461)
- N-step DQN - ```dqn_nstep_agent```
- Vanilla Policy Gradients (VPG/ REINFORCE) - ```vpg_agent```- [paper]()
- Deep Q-learning from Demonstration (DQFD) -
    [paper](https://arxiv.org/abs/1704.03732)
- Proximal Policy Optimisation (PPO) - ```ppp_agent``` - [paper](https://arxiv.org/abs/1707.06347)
- Random and constant agents for sanity checking: ```random_agent```, ```constant_agent```
 
Other heuristics and their respective config key that can be turned on where sensible:

- Generalized advantage estimation - ```gae_lambda```  - [paper](https://arxiv.org/abs/1506.02438)
- Prioritizied experience replay - memory type ```prioritized_replay``` - [paper](https://arxiv.org/abs/1511.05952)
- Bounded continuous actions are mapped to Beta distributions instead of Gaussians - [paper](http://proceedings.mlr.press/v70/chou17a/chou17a.pdf)
- Baseline modes: Based on raw states (```states```) or on network output (```network```). MLP (```mlp```), CNN (```cnn```) or custom network (```custom```). Special case for mode ```states```: baseline per state + linear combination layer (via ```baseline=dict(state1=..., state2=..., etc)```).
- Generic pure TensorFlow optimizers, most models can be used with natural gradient and evolutionary optimizers
- Preprocessing modes: ```normalize```, ```standardize```, ```grayscale```, ```sequence```, ```clip```,
  ```divide```, ```image_resize```
- Exploration modes: ```constant```,```linear_decay```, ```epsilon_anneal```, ```epsilon_decay```,
  ```ornstein_uhlenbeck```

Installation
------------

We uploaded the latest stable version of TensorForce to PyPI. To install, just execute:

```bash
pip install tensorforce
```

If you want to use the latest version from GitHub, use:


```bash
git clone git@github.com:reinforceio/tensorforce.git
cd tensorforce
pip install -e .
```

TensorForce is built on [Google's Tensorflow](https://www.tensorflow.org/). The installation command assumes
that you have `tensorflow` or `tensorflow-gpu` installed.

Alternatively, you can use the following commands to install the tensorflow dependency.

To install TensorForce with `tensorflow` (cpu), use:

```bash
# PyPI install
pip install tensorforce[tf]

# Local install
pip install -e .[tf]
```

To install TensorForce with `tensorflow-gpu` (gpu), use:

```bash
# PyPI install
pip install tensorforce[tf_gpu]

# Local install
pip install -e .[tf_gpu]
```

To update TensorForce, use `pip install --upgrade tensorforce` for the PyPI
version, or run `git pull` in the tensorforce directory if you cloned the 
GitHub repository.
Please note that we did not include OpenAI Gym/Universe/DeepMind lab in
the default install script because not everyone will want to use these.
Please install them as required, usually via pip.

Examples and documentation
--------------------------

For a quick start, you can run one of our example scripts using the
provided configurations, e.g. to run the TRPO agent on CartPole, execute
from the examples folder:

```bash
python examples/openai_gym.py CartPole-v0 -a examples/configs/ppo.json -n examples/configs/mlp2_network.json
```

Documentation is available at
[ReadTheDocs](http://tensorforce.readthedocs.io). We also have tests
validating models on minimal environments which can be run from the main
directory by executing `pytest`{.sourceCode}.

Create and use agents
---------------------

To use TensorForce as a library without using the pre-defined simulation
runners, simply install and import the library, then create an agent and
use it as seen below (see documentation for all optional parameters):

```python
from tensorforce.agents import PPOAgent

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    states_spec=dict(type='float', shape=(10,)),
    actions_spec=dict(type='int', num_actions=10),
    network_spec=[
        dict(type='dense', size=64),
        dict(type='dense', size=64)
    ],
    batch_size=1000,
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-4
    )
)

# Get new data from somewhere, e.g. a client to a web app
client = MyClient('http://127.0.0.1', 8080)

# Poll new state from client
state = client.get_state()

# Get prediction from agent, execute
action = agent.act(state)
reward = client.execute(action)

# Add experience, agent automatically updates model according to batch size
agent.observe(reward=reward, terminal=False)
```

Benchmarks
----------

We provide a seperate repository for benchmarking our algorithm implementations at
[reinforceio/tensorforce-benchmark](https://github.com/reinforceio/tensorforce-benchmark).

Docker containers for benchmarking (CPU and GPU) are available.

This is a sample output for `CartPole-v0`, comparing VPG, TRPO and PPO:

![example output](https://user-images.githubusercontent.com/14904111/29328011-52778284-81f1-11e7-8f70-6554ca9388ed.png)

Please refer to the [tensorforce-benchmark](https://github.com/reinforceio/tensorforce-benchmark) repository
for more information.


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


Community and contributions
---------------------------

TensorForce is developed by [reinforce.io](https://reinforce.io), a new
project focused on providing reinforcement learning software
infrastructure. For any questions, get in touch at
<contact@reinforce.io>.

Please file bug reports and feature discussions as GitHub issues in first instance.

There is also a developer chat you are welcome to join. For joining, we ask to provide
some basic details how you are using TensorForce so we can learn more about applications and our
community. Please fill in [this short form](https://docs.google.com/forms/d/1_UD5Pb5LaPVUviD0pO0fFcEnx_vwenvuc00jmP2rRIc/) which will take
 you to the chat after.

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

We are also very grateful for our open source contributors (listed according to github): Islandman93, wassname, 
trickmeyer, lefnire, mryellow, beflix,AdamStelmaszczyk, 10nagachika, petrbel, Kismuz.
