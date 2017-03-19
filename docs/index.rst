*TensorForce - modular deep reinforcement learning in TensorFlow*
=================================================================================

TensorForce is an open source reinforcement learning library focused on
providing clear APIs, readability and modularisation to deploy
reinforcement learning solutions both in research and practice.
TensorForce is built on top on TensorFlow.

Examples and documentation
--------------------------

For a quick start, you can run one of our example scripts using the
provided configurations, e.g. to run the TRPO agent on CartPole, execute
from the examples folder:

::

    python tensorforce/examples/openai_gym.py CartPole-v0 -a TRPOAgent -c tensorforce/examples/configs/trpo_agent.json -n tensorforce/examples/configs/trpo_network.json
    
You can find some documentation in the `docs <./>`__ directory, such as information about the `agents and models <agents_models.rst>`__, `preprocessing <preprocessing.rst>`__ and the `runners <runner.rst>`__.

More information
----------------

You can find more information in the at our `TensorForce GitHub repository <https://github.com/reinforceio/TensorForce>`__.

