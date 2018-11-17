Preprocessing
=============

Often it is necessary to modify state input tensors before passing them
to the reinforcement learning agent. This could be due to various
reasons, e.g.:

-   Feature scaling / input normalization,
-   Data reduction,
-   Ensuring the Markov property by concatenating multiple states (e.g.
    in Atari)

TensorForce comes with a number of ready-to-use preprocessors, a
preprocessing stack and easy ways to implement your own preprocessors.

Usage
-----

Each preprocessor implements three methods:

1.  The constructor (`__init__`) for parameter initialization
2.  `process(state)` takes a state and returns the processed state
3.  `processed_shape(original_shape)` takes a shape and returns the processed
    shape

The preprocessing stack iteratively calls these functions of all
preprocessors in the stack and returns the result.

### Using one preprocessor

```python
from tensorforce.core.preprocessors import Sequence

pp_seq = Sequence(4)  # initialize preprocessor (return sequence of last 4 states)

state = env.reset()  # reset environment
processed_state = pp_seq.process(state)  # process state
```

### Using a preprocessing stack

You can stack multipe preprocessors:

```python
from tensorforce.core.preprocessors import Preprocessing, Grayscale, Sequence

pp_gray = Grayscale()  # initialize grayscale preprocessor
pp_seq = Sequence(4)  # initialize sequence preprocessor

stack = Preprocessing()  # initialize preprocessing stack
stack.add(pp_gray)  # add grayscale preprocessor to stack
stack.add(pp_seq)  # add maximum preprocessor to stack

state = env.reset()  # reset environment
processed_state = stack.process(state)  # process state
```

### Using a configuration dict

If you use configuration objects, you can build your preprocessing stack
from a config:

```python
from tensorforce.core.preprocessors import Preprocessing

preprocessing_config = [
    {
	    "type": "image_resize",
        "width": 84,
        "height": 84
    }, {
	    "type": "grayscale"
    }, {
	    "type": "center"
    }, {
	    "type": "sequence",
        "length": 4
    }
]

stack = Preprocessing.from_spec(preprocessing_config)
config.state_shape = stack.shape(config.state_shape)
```

The `Agent` class expects a *preprocessing* configuration parameter and then
handles preprocessing automatically:

```
from tensorforce.agents import DQNAgent

agent = DQNAgent(config=dict(
    states=...,
    actions=...,
    preprocessing=preprocessing_config,
    # ...
))
```


Ready-to-use preprocessors
--------------------------

These are the preprocessors that come with TensorForce:

### Standardize

```eval_rst
    .. autoclass:: tensorforce.core.preprocessors.Standardize
        :noindex:
        :show-inheritance:
        :members:
```

### Grayscale

```eval_rst
    .. autoclass:: tensorforce.core.preprocessors.Grayscale
        :noindex:
        :show-inheritance:
        :members:
```

### ImageResize

```eval_rst
    .. autoclass:: tensorforce.core.preprocessors.ImageResize
        :noindex:
        :show-inheritance:
        :members:
```

### Normalize

```eval_rst
    .. autoclass:: tensorforce.core.preprocessors.Normalize
        :noindex:
        :show-inheritance:
        :members:
```

### Sequence

```eval_rst
    .. autoclass:: tensorforce.core.preprocessors.Sequence
        :noindex:
        :show-inheritance:
        :members:
```

Building your own preprocessor
------------------------------

All preprocessors should inherit from
`tensorforce.core.preprocessors.Preprocessor`.

For a start, please refer to the source of the [Grayscale
preprocessor](https://github.com/reinforceio/tensorforce/blob/master/tensorforce/core/preprocessors/grayscale.py).

