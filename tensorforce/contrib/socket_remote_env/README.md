This code allows to wrap the environment through a socket for training through the network. For example, this allows
to run environments on a set of machine, and the Agent on a separate machine.

This code was developped originally in:

```
Rabault, J., Kuhnle, A (2019).
Accelerating Deep Reinforcement Leaning strategies of Flow Control through a
multi-environment approach.
Physics of Fluids.
```

See the article, and the code online, for further help:

```
https://github.com/jerabaul29/Cylinder2DFlowControlDRLParallel
```

NOTE: this is tested only on Ubuntu 18.04

To test, either:

- use our script: ```bash script_launch_parallel.sh  training_example 3000 4```

- launch by hand:

  - ```python3 launch_servers.py -p 3000 -n 4```
  - ```python3 launch_parallel_training.py -p 3000 -n 4```
