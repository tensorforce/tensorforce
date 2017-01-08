# *tensorforce - modular deep reinforcement learning on tensorflow*

## Introduction

tensorforce is an open source reinforcement library focused on providing clear APIs, readability and modularisation to deploy 
reinforcement learning solutions both in research and practice. The main difference to existing libraries
such as rllab is a strict separation of environments, agents and update logic that facilitates usage in non-simulation
environments (see our architecture overview page for details). Further, research code often relies on fixed
network architectures that have been used to tackle particular benchmarks. tensorforce is built with the idea
that (almost) everything should be optionally configurable and in particular uses value function template configurations
to be able to quickly experiment with new models. The overarching goal of tensorforce is to provide a reinforcement
learning framework that smoothly integrates into modern software service architectures via integration with standard open source components.

tensorforce is actively being maintained and developed both to continuously improve the existing code as well as to
reflect new developments as they arise (see roadmap for more). The aim is not to include every new trick as quickly as possible but to
adopt methods as they prove themselves stable, e.g. as of early 2017 A3C and TRPO variants are the basis of a lot
of research.

## Acknowledgements

The goal of tensorforce is not to re-implement existing algorithms, but to provide clear APIs and modularisations,
and later provide serving, integration and deployment components. The credit for original open source implementations, which we have adopted and modified into our architecture, 
fully belongs to the original authors, which have all made their code available under MIT licenses.

In particular, credit goes to John Schulman, Ilya Sutskever and Wojciech Zaremba for their
various TRPO implementations, Rocky Duan for rllab, Taehoon Kim for his DQN and NAF implementations, and many others
who have put in effort to make deep reinforcement learning more accessible through blogposts and 
tutorials.

## Features

tensorforce currently integrates with the OpenAI Gym API, OpenAI universe and DeepMind lab. The following algorithms are available:
1. A3C
2. TRPO with generalised advantage estimation (GAE)
3. Normalised Advantage functions
4. DQN/Double-DQN
5. Vanilla Policy Gradients with GAE

## Installation

## Documentation

## Road map and contributions

tensorforce is still in alpha and hence continuously being updated. Contributions are always welcome, as long as they conform
to our general architecture and code style. We will use github issues to track development. 
The larger roadmap looks as follows:

1. Improve policy gradient internal state management and configurable policies 
2. Execution configuration that abstracts away tensorflow device configurations
3. Generic parallelisation API
4. Hybrid A3C/policy gradient algorithms - not clear yet which combination method will work best, but a 
number of papers showcasing different approaches have been submitted to ICLR 2017.
5. RL serving component. TensorFlow serving can serve trained models but is not suitable to manage RL lifecycles.

## Support and contact

tensorforce is maintained by reinforce.io, a new project focused on providing open source reinforcement learning 
infrastructure. For any questions or support, get in touch at contact@reinforce.io

You are also welcome to join our gitter channel for help with using tensorforce, bugs or contributions:





