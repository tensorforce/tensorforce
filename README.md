# *tensorforce - Modular deep reinforcement learning on top of tensorflow*

## Introduction

tensorforce is an open source reinforcement library focused on providing clear APIs, readability and modularisation to deploy 
reinforcement learning solutions both in research and practice. The main difference to existing libraries
such as rllab is a strict separation of environments, agents and update logic that facilitates usage in non-simulation
environments (see our architecture overview for more details). Further, research code often relies on fixed
network architectures that have been used to tackle particular benchmarks. tensorforce is built with the idea
that (almost) everything should be optionally configurable and in particular uses value function template configurations
to be able to quickly experiment with new models. Going forward, we will aim to provide an interface for straight-forward
parallelisation and tensorflow device mappings. Ultimately, the goal is to make deep reinforcement learning
solutions as easy to deploy as possible in the context of modern compute infrastructures.

tensorforce is actively being maintained and developed both to continuously improve the existing code as well as to
reflect new developments as they arise (see roadmap for more). The aim is not to include every new trick as quickly as possible but to
adopt methods as they prove themselves stable, e.g. as of early 2017 A3C and TRPO variants are the basis of a lot
of research.

## Acknowledgements

The goal of tensorforce is not to re-implement existing algorithms, but to provide clear APIs and modularisations for existing implementations,
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
to our general architecture and code style. We will use github issues to track development. The roadmap looks as follows:

1. Improve policy gradient internal state management and available policies
2. Execution configuration that abstracts away tensorflow device configurations
3. Generic parallelisation API

## Support and contact

tensorforce is maintained by reinforce.io, a new project focused on providing open source reinforcement learning 
infrastructure. For any questions or support, get in touch at contact@reinforce.io

You are also welcome to join our gitter channel for help with using tensorforce, bugs or contributions:





