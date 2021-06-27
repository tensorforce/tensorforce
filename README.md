# Tensorforce: a TensorFlow library for applied reinforcement learning

[![Docs](https://readthedocs.org/projects/tensorforce/badge)](http://tensorforce.readthedocs.io/en/latest/)
[![Gitter](https://badges.gitter.im/tensorforce/community.svg)](https://gitter.im/tensorforce/community)
[![Build Status](https://travis-ci.com/tensorforce/tensorforce.svg?branch=master)](https://travis-ci.com/tensorforce/tensorforce)
[![pypi version](https://img.shields.io/pypi/v/tensorforce)](https://pypi.org/project/Tensorforce/)
[![python version](https://img.shields.io/pypi/pyversions/tensorforce)](https://pypi.org/project/Tensorforce/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/tensorforce/tensorforce/blob/master/LICENSE)
[![Donate](https://img.shields.io/badge/donate-GitHub_Sponsors-yellow)](https://github.com/sponsors/AlexKuhnle)
[![Donate](https://img.shields.io/badge/donate-Liberapay-yellow)](https://liberapay.com/TensorforceTeam/donate)



#### Introduction

Tensorforce is an open-source deep reinforcement learning framework, with an emphasis on modularized flexible library design and straightforward usability for applications in research and practice. Tensorforce is built on top of [Google's TensorFlow framework](https://www.tensorflow.org/) and requires Python 3.

Tensorforce follows a set of high-level design choices which differentiate it from other similar libraries:

- **Modular component-based design**: Feature implementations, above all, strive to be as generally applicable and configurable as possible, potentially at some cost of faithfully resembling details of the introducing paper.
- **Separation of RL algorithm and application**: Algorithms are agnostic to the type and structure of inputs (states/observations) and outputs (actions/decisions), as well as the interaction with the application environment.
- **Full-on TensorFlow models**: The entire reinforcement learning logic, including control flow, is implemented in TensorFlow, to enable portable computation graphs independent of application programming language, and to facilitate the deployment of models.



#### Quicklinks

- [Documentation](http://tensorforce.readthedocs.io) and [update notes](https://github.com/tensorforce/tensorforce/blob/master/UPDATE_NOTES.md)
- [Contact](mailto:tensorforce.team@gmail.com) and [Gitter channel](https://gitter.im/tensorforce/community)
- [Benchmarks](https://github.com/tensorforce/tensorforce/blob/master/benchmarks) and [projects using Tensorforce](https://github.com/tensorforce/tensorforce/blob/master/PROJECTS.md)
- [Roadmap](https://github.com/tensorforce/tensorforce/blob/master/ROADMAP.md) and [contribution guidelines](https://github.com/tensorforce/tensorforce/blob/master/CONTRIBUTING.md)
- [GitHub Sponsors](https://github.com/sponsors/AlexKuhnle) and [Liberapay](https://liberapay.com/TensorforceTeam/donate)



#### Table of content

- [Installation](#installation)
- [Quickstart example code](#quickstart-example-code)
- [Command line usage](#command-line-usage)
- [Features](#features)
- [Environment adapters](#environment-adapters)
- [Support, feedback and donating](#support-feedback-and-donating)
- [Core team and contributors](#core-team-and-contributors)
- [Cite Tensorforce](#cite-tensorforce)



## Installation

A stable version of Tensorforce is periodically updated on PyPI and installed as follows:

```bash
pip3 install tensorforce
```

To always use the latest version of Tensorforce, install the GitHub version instead:

```bash
git clone https://github.com/tensorforce/tensorforce.git
pip3 install -e tensorforce
```

**Note on installation on M1 Macs:** At the moment Tensorflow, which is a core dependency of Tensorforce, cannot be installed on M1 Macs directly. Follow the ["M1 Macs" section](https://tensorforce.readthedocs.io/en/latest/basics/installation.html) in the documentation for a workaround.

Environments require additional packages for which there are setup options available (`ale`, `gym`, `retro`, `vizdoom`, `carla`; or `envs` for all environments), however, some require additional tools to be installed separately (see [environments documentation](http://tensorforce.readthedocs.io)). Other setup options include `tfa` for [TensorFlow Addons](https://www.tensorflow.org/addons) and `tune` for [HpBandSter](https://github.com/automl/HpBandSter) required for the `tune.py` script.

**Note on GPU usage:** Different from (un)supervised deep learning, RL does not always benefit from running on a GPU, depending on environment and agent configuration. In particular for environments with low-dimensional state spaces (i.e., no images), it is hence worth trying to run on CPU only.



## Quickstart example code

```python
from tensorforce import Agent, Environment

# Pre-defined or custom environment
environment = Environment.create(
    environment='gym', level='CartPole', max_episode_timesteps=500
)

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='tensorforce',
    environment=environment,  # alternatively: states, actions, (max_episode_timesteps)
    memory=10000,
    update=dict(unit='timesteps', batch_size=64),
    optimizer=dict(type='adam', learning_rate=3e-4),
    policy=dict(network='auto'),
    objective='policy_gradient',
    reward_estimation=dict(horizon=20)
)

# Train for 300 episodes
for _ in range(300):

    # Initialize episode
    states = environment.reset()
    terminal = False

    while not terminal:
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

agent.close()
environment.close()
```



## Command line usage

Tensorforce comes with a range of [example configurations](https://github.com/tensorforce/tensorforce/tree/master/benchmarks/configs) for different popular reinforcement learning environments. For instance, to run Tensorforce's implementation of the popular [Proximal Policy Optimization (PPO) algorithm](https://arxiv.org/abs/1707.06347) on the [OpenAI Gym CartPole environment](https://gym.openai.com/envs/CartPole-v1/), execute the following line:

```bash
python3 run.py --agent benchmarks/configs/ppo.json --environment gym \
    --level CartPole-v1 --episodes 100
```

For more information check out the [documentation](http://tensorforce.readthedocs.io).



## Features

- **Network layers**: Fully-connected, 1- and 2-dimensional convolutions, embeddings, pooling, RNNs, dropout, normalization, and more; *plus* support of Keras layers.
- **Network architecture**: Support for multi-state inputs and layer (block) reuse, simple definition of directed acyclic graph structures via register/retrieve layer, plus support for arbitrary architectures.
- **Memory types**: Simple batch buffer memory, random replay memory.
- **Policy distributions**: Bernoulli distribution for boolean actions, categorical distribution for (finite) integer actions, Gaussian distribution for continuous actions, Beta distribution for range-constrained continuous actions, multi-action support.
- **Reward estimation**: Configuration options for estimation horizon, future reward discount, state/state-action/advantage estimation, and for whether to consider terminal and horizon states.
- **Training objectives**: (Deterministic) policy gradient, state-(action-)value approximation.
- **Optimization algorithms**: Various gradient-based optimizers provided by TensorFlow like Adam/AdaDelta/RMSProp/etc, evolutionary optimizer, natural-gradient-based optimizer, plus a range of meta-optimizers.
- **Exploration**: Randomized actions, sampling temperature, variable noise.
- **Preprocessing**: Clipping, deltafier, sequence, image processing.
- **Regularization**: L2 and entropy regularization.
- **Execution modes**: Parallelized execution of multiple environments based on Python's `multiprocessing` and `socket`.
- **Optimized act-only SavedModel extraction**.
- **TensorBoard support**.

By combining these modular components in different ways, a variety of popular deep reinforcement learning models/features can be replicated:

- Q-learning: [Deep Q-learning](https://www.nature.com/articles/nature14236), [Double-DQN](https://arxiv.org/abs/1509.06461), [Dueling DQN](https://arxiv.org/abs/1511.06581), [n-step DQN](https://arxiv.org/abs/1602.01783), [Normalised Advantage Function (NAF)](https://arxiv.org/abs/1603.00748)
- Policy gradient: [vanilla policy-gradient / REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf), [Actor-critic and A3C](https://arxiv.org/abs/1602.01783), [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347), [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477), [Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971)

Note that in general the replication is not 100% faithful, since the models as described in the corresponding paper often involve additional minor tweaks and modifications which are hard to support with a modular design (and, arguably, also questionable whether it is important/desirable to support them). On the upside, these models are just a few examples from the multitude of module combinations supported by Tensorforce.



## Environment adapters

- [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment), a simple object-oriented framework that allows researchers and hobbyists to develop AI agents for Atari 2600 games.
- [CARLA](https://github.com/carla-simulator/carla), is an open-source simulator for autonomous driving research.
- [OpenAI Gym](https://gym.openai.com/), a toolkit for developing and comparing reinforcement learning algorithms which supports teaching agents everything from walking to playing games like Pong or Pinball.
- [OpenAI Retro](https://github.com/openai/retro), lets you turn classic video games into Gym environments for reinforcement learning and comes with integrations for ~1000 games.
- [OpenSim](http://osim-rl.stanford.edu/), reinforcement learning with musculoskeletal models.
- [PyGame Learning Environment](https://github.com/ntasfi/PyGame-Learning-Environment/), learning environment which allows a quick start to Reinforcement Learning in Python.
- [ViZDoom](https://github.com/mwydmuch/ViZDoom), allows developing AI bots that play Doom using only the visual information.


## Support, feedback and donating

Please get in touch via [mail](mailto:tensorforce.team@gmail.com) or on [Gitter](https://gitter.im/tensorforce/community) if you have questions, feedback, ideas for features/collaboration, or if you seek support for applying Tensorforce to your problem.

If you want to support the Tensorforce core team (see below), please also consider donating: [GitHub Sponsors](https://github.com/sponsors/AlexKuhnle) or [Liberapay](https://liberapay.com/TensorforceTeam/donate).



## Core team and contributors

Tensorforce is currently developed and maintained by [Alexander Kuhnle](https://github.com/AlexKuhnle).

Earlier versions of Tensorforce (<= 0.4.2) were developed by [Michael Schaarschmidt](https://github.com/michaelschaarschmidt), [Alexander Kuhnle](https://github.com/AlexKuhnle) and [Kai Fricke](https://github.com/krfricke).

The advanced parallel execution functionality was originally contributed by Jean Rabault (@jerabaul29) and Vincent Belus (@vbelus). Moreover, the pretraining feature was largely developed in collaboration with Hongwei Tang (@thw1021) and Jean Rabault (@jerabaul29).

The CARLA environment wrapper is currently developed by Luca Anzalone (@luca96).

We are very grateful for our open-source contributors (listed according to Github, updated periodically):

Islandman93, sven1977, Mazecreator, wassname, lefnire, daggertye, trickmeyer, mkempers,
mryellow, ImpulseAdventure,
janislavjankov, andrewekhalel,
HassamSheikh, skervim,
beflix, coord-e,
benelot, tms1337, vwxyzjn, erniejunior,
Deathn0t, petrbel, nrhodes, batu, yellowbee686, tgianko,
AdamStelmaszczyk, BorisSchaeling, christianhidber, Davidnet, ekerazha, gitter-badger, kborozdin, Kismuz, mannsi, milesmcc, nagachika, neitzal, ngoodger, perara, sohakes, tomhennigan.



## Cite Tensorforce

Please cite the framework as follows:

```
@misc{tensorforce,
  author       = {Kuhnle, Alexander and Schaarschmidt, Michael and Fricke, Kai},
  title        = {Tensorforce: a TensorFlow library for applied reinforcement learning},
  howpublished = {Web page},
  url          = {https://github.com/tensorforce/tensorforce},
  year         = {2017}
}
```

If you use the [parallel execution functionality](https://github.com/tensorforce/tensorforce/tree/master/tensorforce/contrib), please additionally cite it as follows:

```
@article{rabault2019accelerating,
  title        = {Accelerating deep reinforcement learning strategies of flow control through a multi-environment approach},
  author       = {Rabault, Jean and Kuhnle, Alexander},
  journal      = {Physics of Fluids},
  volume       = {31},
  number       = {9},
  pages        = {094105},
  year         = {2019},
  publisher    = {AIP Publishing}
}
```

If you use Tensorforce in your research, you may additionally consider citing the following paper:

```
@article{lift-tensorforce,
  author       = {Schaarschmidt, Michael and Kuhnle, Alexander and Ellis, Ben and Fricke, Kai and Gessert, Felix and Yoneki, Eiko},
  title        = {{LIFT}: Reinforcement Learning in Computer Systems by Learning From Demonstrations},
  journal      = {CoRR},
  volume       = {abs/1808.07903},
  year         = {2018},
  url          = {http://arxiv.org/abs/1808.07903},
  archivePrefix = {arXiv},
  eprint       = {1808.07903}
}
```
