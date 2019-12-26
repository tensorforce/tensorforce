# Tensorforce: a TensorFlow library for applied reinforcement learning

[![Docs](https://readthedocs.org/projects/tensorforce/badge)](http://tensorforce.readthedocs.io/en/latest/)
[![Gitter](https://badges.gitter.im/tensorforce/community.svg)](https://gitter.im/tensorforce/community)
[![Build Status](https://travis-ci.com/tensorforce/tensorforce.svg?branch=master)](https://travis-ci.com/tensorforce/tensorforce)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/tensorforce/tensorforce/blob/master/LICENSE)



#### Introduction

Tensorforce is an open-source deep reinforcement learning framework, with an emphasis on modularized flexible library design and straightforward usability for applications in research and practice. Tensorforce is built on top of [Google's TensorFlow framework](https://www.tensorflow.org/) and compatible with Python 3 (Python 2 support was dropped with version 0.5).

Tensorforce follows a set of high-level design choices which differentiate it from other similar libraries:

- **Modular component-based design**: Feature implementations, above all, strive to be as generally applicable and configurable as possible, potentially at some cost of faithfully resembling details of the introducing paper.
- **Separation of RL algorithm and application**: Algorithms are agnostic to the type and structure of inputs (states/observations) and outputs (actions/decisions), as well as the interaction with the application environment.
- **Full-on TensorFlow models**: The entire reinforcement learning logic, including control flow, is implemented in TensorFlow, to enable portable computation graphs independent of application programming language, and to facilitate the deployment of models.



#### Quicklinks

- [Documentation](http://tensorforce.readthedocs.io) and [update notes](https://github.com/tensorforce/tensorforce/blob/master/UPDATE_NOTES.md)
- [Benchmarks](https://github.com/tensorforce/tensorforce/blob/master/benchmarks) and [projects using Tensorforce](https://github.com/tensorforce/tensorforce/blob/master/PROJECTS.md)
- [Contact](mailto:tensorforce.team@gmail.com) and [Gitter channel](https://gitter.im/tensorforce/community)
- [Roadmap](https://github.com/tensorforce/tensorforce/blob/master/ROADMAP.md) and [contribution guidelines](https://github.com/tensorforce/tensorforce/blob/master/CONTRIBUTING.md)



#### Table of content

- [Installation](#installation)
- [Quickstart example code](#quickstart-example-code)
- [Command line usage](#command-line-usage)
- [Features](#features)
- [Environment adapters](#environment-adapters)
- [Contact for support and feedback](#contact-for-support-and-feedback)
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
cd tensorforce
pip3 install -e .
```

Tensorforce is built on top of [Google's TensorFlow](https://www.tensorflow.org/) and requires that either `tensorflow` or `tensorflow-gpu` is installed. To include the correct version of TensorFlow with the installation of Tensorforce, simply add the flag `tf` for the normal CPU version or `tf_gpu` for the GPU version:

```bash
# PyPI version plus TensorFlow CPU version
pip3 install tensorforce[tf]

# GitHub version plus TensorFlow GPU version
pip3 install -e .[tf_gpu]
```

Some environments require additional packages, for which there are also options available (`mazeexp`, `gym`, `retro`, `vizdoom`; or `envs` for all environments), however, some require other tools to be installed (see [environments documentation](http://tensorforce.readthedocs.io)).



## Quickstart example code

```python
from tensorforce.agents import Agent

# Instantiate a Tensorforce agent
agent = Agent.create(
    agent='tensorforce',
    states=dict(type='float', shape=(10,)),
    actions=dict(type='int', num_values=5),
    max_episode_timesteps=100,
    memory=10000,
    update=dict(unit='timesteps', batch_size=64),
    optimizer=dict(type='adam', learning_rate=3e-4),
    policy=dict(network='auto'),
    objective='policy_gradient',
    reward_estimation=dict(horizon=20)
)

# Retrieve the latest (observable) environment state
state = get_current_state()  # (float array of shape [10])

# Query the agent for its action decision
action = agent.act(states=state)  # (scalar between 0 and 4)

# Execute the decision and retrieve the current performance score
reward = execute_decision(action)  # (any scalar float)

# Pass feedback about performance (and termination) to the agent
agent.observe(reward=reward, terminal=False)
```



## Command line usage

Tensorforce comes with a range of [example configurations](https://github.com/tensorforce/tensorforce/tree/master/benchmarks/configs) for different popular reinforcement learning environments. For instance, to run Tensorforce's implementation of the popular [Proximal Policy Optimization (PPO) algorithm](https://arxiv.org/abs/1707.06347) on the [OpenAI Gym CartPole environment](https://gym.openai.com/envs/CartPole-v1/), execute the following line:

```bash
python3 run.py benchmarks/configs/ppo.json gym --level CartPole-v1 -e 300
```

For more information check out the [documentation](http://tensorforce.readthedocs.io).



## Features

- **Neural network layers**: Dense fully-connected layer, embedding layer, 1- and 2-dimensional convolution, pooling, LSTM, activation, dropout, normalization, and more; *plus* support of Keras layers.
- **Memory types**: Simple batch buffer memory, random replay memory.
- **Policy distributions**: Bernoulli distribution for boolean actions, categorical distribution for (finite) integer actions, Gaussian distribution for continuous actions, Beta distribution for range-constrained continuous actions.
- **Optimization algorithms**: Various gradient-based optimizers provided by TensorFlow like Adam/AdaDelta/Momentum/RMSProp/etc, evolutionary optimizer, natural-gradient-based optimizer, plus a range of meta-optimizers.
- **Execution modes**: Parallel execution, distributed execution.
- **Other features**: state/reward preprocessing, exploration, variable noise, regularization losses.
- **TensorBoard support**.

By combining these modular components in different ways, a variety of popular deep reinforcement learning models/features can be replicated: [Deep Q-learning (DQN)](https://arxiv.org/abs/1312.5602) and variations like [Double-DQN](https://arxiv.org/abs/1509.06461) or [Deep Q-learning from Demonstrations (DQfD)](https://arxiv.org/abs/1704.03732), [vanilla policy-gradient algorithm / REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf), [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347), [Actor-critic and A3C](https://arxiv.org/abs/1602.01783), [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477), [Normalised Advantage Function (NAF)](https://arxiv.org/abs/1603.00748), [Generalized Advantage estimation (GAE)](https://arxiv.org/abs/1506.02438), etc.

Note that in general the replication is not 100% faithful, since the models as described in the corresponding paper often involve additional minor tweaks and modifications which are hard to support with a modular design (and, arguably, also questionable whether it is important/desirable to support them). On the upside, these models are just a few examples from the multitude of module combinations supported by Tensorforce.



## Environment adapters

- [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment), a simple object-oriented framework that allows researchers and hobbyists to develop AI agents for Atari 2600 games.
- [MazeExplorer](https://github.com/mryellow/maze_explorer), a maze exploration game for AI agents.
- [OpenAI Gym](https://gym.openai.com/), a toolkit for developing and comparing reinforcement learning algorithms which supports teaching agents everything from walking to playing games like Pong or Pinball.
- [OpenAI Retro](https://github.com/openai/retro), lets you turn classic video games into Gym environments for reinforcement learning and comes with integrations for ~1000 games.
- [OpenSim](http://osim-rl.stanford.edu/), reinforcement learning with musculoskeletal models.
- [PyGame Learning Environment](https://github.com/ntasfi/PyGame-Learning-Environment/), learning environment which allows a quick start to Reinforcement Learning in Python.
- [ViZDoom](https://github.com/mwydmuch/ViZDoom), allows developing AI bots that play Doom using only the visual information.



## Contact for support and feedback

Please get in touch via [mail](mailto:tensorforce.team@gmail.com) or on [Gitter](https://gitter.im/tensorforce/community) if you have questions, feedback, ideas for features/collaboration, or if you seek support for applying Tensorforce to your problem.



## Core team and contributors

Tensorforce is currently developed and maintained by [Alexander Kuhnle](https://github.com/AlexKuhnle). Earlier versions of Tensorforce (<= 0.4.2) were developed by [Michael Schaarschmidt](https://github.com/michaelschaarschmidt), [Alexander Kuhnle](https://github.com/AlexKuhnle) and [Kai Fricke](https://github.com/krfricke).

The advanced parallel execution functionality (currently in [this module](https://github.com/tensorforce/tensorforce/tree/master/tensorforce/contrib)) was contributed by Jean Rabault (@jerabaul29). Moreover, the pretraining feature was largely developed in collaboration with Hongwei Tang (@thw1021) and Jean Rabault (@jerabaul29).

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
