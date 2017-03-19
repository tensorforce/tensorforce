For a quick start, you can run one of our example scripts using the
provided configurations, e.g. to run the TRPO agent on CartPole, execute
from the examples folder:

::

    python tensorforce/examples/openai_gym.py CartPole-v0 -a TRPOAgent -c tensorforce/examples/configs/trpo_agent.json -n tensorforce/examples/configs/trpo_network.json
    
You can find some documentation in the `docs <./>`__ directory, such as information about the `agents and models <agents_models.rst>`__, `preprocessing <preprocessing.rst>`__ and the `runners <runner.rst>`__.
