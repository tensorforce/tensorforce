*TensorForce - modular deep reinforcement learning in TensorFlow*
=================================================================================

TensorForce is an open source reinforcement learning library focused on
providing clear APIs, readability and modularisation to deploy
reinforcement learning solutions both in research and practice.
TensorForce is built on top on TensorFlow.

Quick start
-----------

For a quick start, you can run one of our example scripts using the
provided configurations, e.g. to run the TRPO agent on CartPole, execute
from the examples folder:

.. code:: bash

    python examples/openai_gym.py CartPole-v0 -a examples/configs/ppo.json -n examples/configs/mlp2_network.json


In python, it could look like this:

.. code:: python

    # examples/quickstart.py

   import numpy as np

   from tensorforce.agents import PPOAgent
   from tensorforce.execution import Runner
   from tensorforce.contrib.openai_gym import OpenAIGym

   # Create an OpenAIgym environment
   env = OpenAIGym('CartPole-v0', visualize=True)

   # Network as list of layers
   network_spec = [
       dict(type='dense', size=32, activation='tanh'),
       dict(type='dense', size=32, activation='tanh')
   ]

   agent = PPOAgent(
       states_spec=env.states,
       actions_spec=env.actions,
       network_spec=network_spec,
       batch_size=4096,
       # BatchAgent
       keep_last_timestep=True,
       # PPOAgent
       step_optimizer=dict(
           type='adam',
           learning_rate=1e-3
       ),
       optimization_steps=10,
       # Model
       scope='ppo',
       discount=0.99,
       # DistributionModel
       distributions_spec=None,
       entropy_regularization=0.01,
       # PGModel
       baseline_mode=None,
       baseline=None,
       baseline_optimizer=None,
       gae_lambda=None,
       # PGLRModel
       likelihood_ratio_clipping=0.2,
       summary_spec=None,
       distributed_spec=None
   )

   # Create the runner
   runner = Runner(agent=agent, environment=env)


   # Callback function printing episode statistics
   def episode_finished(r):
       print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                    reward=r.episode_rewards[-1]))
       return True


   # Start learning
   runner.run(episodes=3000, max_episode_timesteps=200, episode_finished=episode_finished)
   runner.close()

   # Print statistics
   print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
       ep=runner.episode,
       ar=np.mean(runner.episode_rewards[-100:]))
   )


.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   agents_models
   environments
   preprocessing
   summary_spec
   runner
   tensorforce/tensorforce


More information
----------------

You can find more information at our `TensorForce GitHub repository <https://github.com/reinforceio/TensorForce>`__.

We have a seperate repository available for benchmarking our algorithm implementations
[here](https://github.com/reinforceio/tensorforce-benchmark).
