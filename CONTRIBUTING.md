Contribution guide
------------------

Below are some pointers for new contributions. In general, it is probably always a good idea to
join the community to discuss a contribution, unless it's a smaller bug fix. You can join the
community by filling in
[this short form](https://docs.google.com/forms/d/1_UD5Pb5LaPVUviD0pO0fFcEnx_vwenvuc00jmP2rRIc/)
which will take you to the chat after.

### 1. Code style

In general, we try to follow the
[Google Python style guide](https://google.github.io/styleguide/pyguide.html) with a few
particulars. Another good rule of thumb is that if something is a PEP8 warning in your editor, it
is probably worth looking at.

Some things to pay attention to:

- Lines should have a max length of 120 characters, 100 for documentation comments.

- When initializing objects such as dictionaries or lists and there are multiple entries, use the
following format:

```python
# One key-value pair per line, one indent level.
dict(
    states=states,
    internals=internals,
    actions=actions,
    terminal=terminal,
    reward=reward
)
```

- When calling TensorFlow functions, use named arguments for readability wherever possible:

```python
scaffold = tf.train.Scaffold(
    init_op=init_op,
    init_feed_dict=None,
    init_fn=init_fn,
    ready_op=ready_op,
    ready_for_local_init_op=ready_for_local_init_op,
    local_init_op=local_init_op,
    summary_op=summary_op,
    saver=saver,
    copy_from_scaffold=None
)
```

- Indentations should always be tab-spaced (tab size: 4), instead of based on alignments to the previous line:

```python
states_preprocessing_variables = [
    variable for name in self.states_preprocessing.keys()
    for variable in self.states_preprocessing[name].get_variables()
]
```

instead of:

```python
states_preprocessing_variables = [variable for name in self.states_preprocessing.keys()
                                  for variable in self.states_preprocessing[name].get_variables()]
```

or:

```python
kwargs['fn_loss'] = (lambda: self.fn_loss(
    states=states,
    internals=internals,
    actions=actions,
    terminal=terminal,
    reward=reward,
    update=update
))
```

instead of:

```python
kwargs['fn_loss'] = (
    lambda: self.fn_loss(states=states, internals=internals, actions=actions,
                         terminal=terminal, reward=reward, update=update)
)
```

- Binary operators should always be surrounded by a single space: `z = x + y` instead of `z=x+y`.

- Numbers should always be explicitly given according to their intended type, so floats always with period, `1.0`, and integers without, `1`. Floats should furthermore explicitly add single leading/trailing zeros where applicable, so `1.0` instead of `1.` and `0.1` instead of `.1`.

- Lines (apart from documentation comments), including empty lines, should never contain trailing
spaces.

- Comments, even line comments, should be capitalised.

- We prefer line comments to be above the line of code they are commenting on for shorter lines:

```python
# This is a line comment.
input_values = dict()
```

instead of:

```python
input_values = dict()  # this is a non-capitalised line comment making the line unnecessarily long
```


### 2. Architecture

New contributions should integrate into the existing design ideas. To this end, reading our
[blog](https://reinforce.io/blog/)) can be very helpful. The key design elements to understand are
the optimizer package (as described in the blog), the idea to move all reinforcement learning
control flow into the TensorFlow graph, and the general object hierarchy of models. Again, for
detailed questions do join the chat.


### 3. Areas of contribution

Below are some potential areas of contribution. Feel free to make new suggestions on your own.

Environments: 

TensorForce provides a generic
[enviroment class](https://github.com/reinforceio/tensorforce/blob/master/tensorforce/environments/environment.py).
Applications do not need to implement this but it provides the advantage of using tne ```Runner```
execution infrastructure. A number of implemented environments can be found in the contrib folder.
Implementing a binding for a new environment is a great way to better understand the agent API and
makes for a good first contribution. Below is a list of environments one might look at:

- Gazebo robotic simulation - [link](http://gazebosim.org)
- Carla, Open-source simulator for autonomous driving research - [link](https://github.com/carla-simulator/carla)
- Unity game engine - [link](https://github.com/Unity-Technologies/ml-agents)
- Project Malmo minecraft bindnig - [link](https://github.com/Microsoft/malmo)
- DeepMind Starcraft 2 learning environment - [link](https://github.com/deepmind/pysc2)
- DeepMind control, dm_control - [link](https://github.com/deepmind/dm_control)
- DeepMind pycolab - [link](https://github.com/deepmind/pycolab)
- OpenAI roboschool - [link](https://github.com/openai/roboschool)
- DeepGTAV - GTA 5 self-driving car research environment - [link](https://github.com/aitorzip/DeepGTAV)
- Siemens industrial control benchmark - [link](https://github.com/siemens/industrialbenchmark)

Models:

Reinforcement learning is a highly active field of research and new models are appearing with high
frequency. Our main development focus is on providing abstractions and architecture, so model
contributions are very welcome. Note that integrating some models may require some discussion on
interfacing the existing models, especially in the case of newer architectures with complex
internal models. Some model suggestions:

- ACER - [paper](https://arxiv.org/abs/1611.01224)
- Direct future prediction (n.b. this will require architecture changes) - [paper](https://arxiv.org/abs/1611.01779)
- Categorical DQN reimplementation. A categorical DQN implementation was part of 0.2.2 but was removed
  because it did not easily integrate into the optimizer architecture. If you are interested in this model,
  please comment in the issue or join the chat for discussion.
- Rainbow DQN, needs categorical DQN first. - [paper](https://arxiv.org/abs/1710.02298)
- Distributional RL with quantile regression - [paper](https://arxiv.org/pdf/1710.10044.pdf)

Ecosystem integrations:

If you are interested in general usability, another area of contribution is integrations into the
wider machine learning and data processing ecosystem. For example, providing scripts to run
TensorForce on one of a number of cloud service providers, or to run jobs on data infrastructure
frameworks like Kubernetes, Spark, etc is a great way to make RL more accessible. Note
that there exists now a Kubernetes client project which can be used as a starting point
for more work.
