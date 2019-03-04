# Tensorforce: a TensorFlow library for applied reinforcement learning

[![Docs](https://readthedocs.org/projects/tensorforce/badge)](http://tensorforce.readthedocs.io/en/latest/)
[![Gitter](https://badges.gitter.im/tensorforce/community.svg)](https://gitter.im/tensorforce/community)
[![Build Status](https://travis-ci.org/tensorforce/tensorforce.svg?branch=master)](https://travis-ci.org/tensorforce/tensorforce)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/tensorforce/tensorforce/blob/master/LICENSE)


**Important**: Currently working on a major revision of the framework, which fixes a lot of internal problems and introduces a range of new features. You can check it out under the [major-revision branch](https://github.com/tensorforce/tensorforce/tree/major-revision). As it's still work in progress, please post an issue or get in touch via [Gitter](https://gitter.im/tensorforce/community) or [mail](mailto:tensorforce.team@gmail.com) if you encounter problems.


**Important**: Tensorforce was recently moved to another GitHub host organization. The following command will update your git directory (assuming no changes beyond standard cloning):

```bash
git remote set-url origin https://github.com/tensorforce/tensorforce.git
```



#### Introduction

Tensorforce is an open-source deep reinforcement learning framework, with an emphasis on modularized flexible library design and straightforward usability for applications in research and practice. TensorForce is built on top of [Google's TensorFlow framework](https://www.tensorflow.org/) and compatible with Python 3 (Python 2 support was dropped with version 0.5).

Tensorforce follows a set of high-level design choices which differentiate it from other similar libraries:

- **Modular component-based design**: Feature implementations, above all, strive to be as generally applicable and configurable as possible, potentially at some cost of faithfully resembling details of the introducing paper.
- **Separation of RL algorithm and application**: Algorithms are agnostic to the type and structure of inputs (states/observations) and outputs (actions/decisions), as well as the interaction with the application environment.
- **Full-on TensorFlow models**: The entire reinforcement learning logic, including control flow, is implemented in TensorFlow, to enable portable computation graphs independent of application programming language, and to facilitate the deployment of models.



#### Quicklinks

- [Documentation](http://tensorforce.readthedocs.io) and [update notes](https://github.com/tensorforce/tensorforce/blob/master/UPDATE_NOTES.md)
- [Contact](mailto:tensorforce.team@gmail.com) and [Gitter channel](https://gitter.im/tensorforce/community)
- [Contribution guidelines](https://github.com/tensorforce/tensorforce/blob/master/CONTRIBUTING.md)
- [Blog](https://reinforce.io/blog/)



#### Table of content

- [Installation](#installation)
- [Features](#features)
- [Examples and documentation](#examples-and-documentation)
- [Quickstart example code](#quickstart-example-code)
- [Contact for support and feedback](#contact-for-support=and-feedback)
- [Core team and contributors](#core-team-and-contributors)
- [Cite Tensorforce](#cite-tensorforce)



## Installation


A stable version of Tensorforce is periodically updated on PyPI and installed as follows:

```bash
pip install tensorforce
```

To always use the latest version of Tensorforce, install the GitHub version instead:

```bash
git clone https://github.com/tensorforce/tensorforce.git
cd tensorforce
pip install -e .
```

TensorForce is built on [Google's TensorFlow](https://www.tensorflow.org/) and requires that either `tensorflow` or `tensorflow-gpu` is installed. Generally, Tensorforce assumes the latest version of TensorFlow and thus is only backwards-compatible to the degree TensorFlow is. To include the current version of TensorFlow with the installation of Tensorforce, add the flag `tf` for the normal CPU version or `tf_gpu` for the GPU version:

```bash
# PyPI version plus TensorFlow CPU version
pip install tensorforce[tf]

# GitHub version plus TensorFlow GPU version
pip install -e .[tf_gpu]
```

Some scripts require additional packages like, for instance, [OpenAI Gym](https://gym.openai.com/), which have to be installed separately.



## Quickstart example code

```python
from tensorforce.agents import PPOAgent

# Instantiate a Tensorforce agent
agent = PPOAgent(
    states=dict(type='float', shape=(10,)),
    actions=dict(type='int', num_values=5),
    network=[
        dict(type='dense', size=64),
        dict(type='dense', size=64)
    ],
    step_optimizer=dict(type='adam', learning_rate=1e-4)
)

# Initialize the agent
agent.initialize()

# Retrieve the latest (observable) environment state
state = get_current_state()  # (float array of shape [10])

# Query the agent for its action decision
action = agent.act(states=state)  # (scalar between 0 and 4)

# Execute the decision and retrieve the current performance score
reward = execute_decision(action)  # (any scalar float)

# Pass feedback about performance (and termination) to the agent
agent.observe(reward=reward, terminal=False)
```



## Features

- **Neural network layers**: Dense fully-connected layer, embedding layer, 1- and 2-dimensional convolution, LSTM, activation layer, dropout.
- **Memory types**: Simple batch buffer memory, random replay memory.
- **Policy distributions**: Bernoulli distribution for boolean actions, categorical distribution for (finite) integer actions, Gaussian distribution for continuous actions, Beta distribution for range-constrained continuous actions.
- **Optimization algorithms**: Various gradient-based optimizers provided by TensorFlow like Adam/AdaDelta/Momentum/RMSProp/etc, evolutionary optimizer, natural-gradient-based optimizer, plus a variety of meta-optimizer.
- **Execution modes**: Parallel execution, distributed execution.
- **Other features**: Input normalization preprocessing, exploration, variable noise, regularization losses.
- **TensorBoard support**.

By combining these modular components in different ways, a variety of popular deep reinforcement learning models/features can be replicated: [Deep Q-learning (DQN)](https://arxiv.org/abs/1312.5602) and variations like [Double-DQN](https://arxiv.org/abs/1509.06461) or [Deep Q-learning from Demonstrations (DQfD)](https://arxiv.org/abs/1704.03732), [vanilla policy-gradient algorithm / REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf), [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347), [Actor-critic and A3C](https://arxiv.org/abs/1602.01783), [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477), [Normalised Advantage Function (NAF)](https://arxiv.org/abs/1603.00748), [Generalized Advantage estimation (GAE)](https://arxiv.org/abs/1506.02438), etc.

Note that in general the replication is not 100% faithful, since the models as described in the corresponding paper often involve additional minor tweaks and modifications which are hard to support with a modular design (and, arguably, also questionable whether it is important/desirable to support them). On the upside, these models are just a few examples from the multitude of module combinations supported by Tensorforce.



## Examples and documentation

Tensorforce comes with a range of [example scripts and configurations](https://github.com/tensorforce/tensorforce/tree/master/examples) for different popular reinforcement learning environments/benchmarks. For instance, to run Tensorforce's implementation of the popular [Proximal Policy Optimization (PPO) algorithm](https://arxiv.org/abs/1707.06347) on the [OpenAI Gym CartPole environment](https://gym.openai.com/envs/CartPole-v1/), execute the following line:

```bash
python examples/openai_gym.py CartPole-v1 --agent examples/configs/ppo.json --network examples/configs/mlp2_network.json
```

For more information check out [Tensorforce's documentation](http://tensorforce.readthedocs.io).



## Contact for support and feedback

Please get in touch via [mail](mailto:tensorforce.team@gmail.com) or on [Gitter](https://gitter.im/tensorforce/community) if you have questions, feedback, ideas for features/collaboration, or if you seek support for applying Tensorforce to your problem.



## Core team and contributors

Tensorforce is currently developed and maintained by [Alexander Kuhnle](https://github.com/AlexKuhnle). Earlier versions of Tensorforce (<= 0.4.2) were developed by [Michael Schaarschmidt](https://github.com/michaelschaarschmidt), [Alexander Kuhnle](https://github.com/AlexKuhnle) and [Kai Fricke](https://github.com/krfricke).

We are very grateful for our open-source contributors (listed according to Github, updated periodically):

Islandman93, sven1977, Mazecreator, wassname, lefnire, daggertye, trickmeyer, mkempers,
mryellow, ImpulseAdventure,
janislavjankov, andrewekhalel,
HassamSheikh, skervim,
beflix, coord-e,
benelot, tms1337, vwxyzjn, erniejunior,
vermashresth, Deathn0t, petrbel, nrhodes, batu, yellowbee686, tgianko,
AdamStelmaszczyk, mannsi, perara, neitzal, gitter-badger, sohakes, ekerazha, nagachika, Davidnet, Kismuz, ngoodger, BorisSchaeling, tomhennigan.



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
