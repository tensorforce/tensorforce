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
3.  `shape(original_shape)` takes a shape and returns the processed
    shape

The preprocessing stack iteratively calls these functions of all
preprocessors in the stack and returns the result.

### Using one preprocessor

```python
from tensorforce.core.preprocessing import Maximum

pp_max = Maximum(2)  # initialize preprocessor (return maximum of last 2 states)

state = env.reset()  # reset environment
processed_state = pp_max.process(state)  # process state
```

### Using a preprocessing stack

You can stack multipe preprocessors:

```python
from tensorforce.preprocessing import Stack, Grayscale, Maximum

pp_gray = Grayscale()  # initialize grayscale preprocessor
pp_max = Maximum(2)  # initialize maximum preprocessor (return maximum of last 2 states)

stack = Stack()  # initialize preprocessing stack
stack += pp_gray  # add grayscale preprocessor to stack
stack += pp_max  # add maximum preprocessor to stack

state = env.reset()  # reset environment
processed_state = stack.process(state)  # process state
```

### Using a configuration dict

If you use configuration objects, you can build your preprocessing stack
from a config:

```python
from tensorforce.util.experiment_util import build_preprocessing_stack

preprocessing_config = {
  "preprocessing": [
    ["grayscale"],
    ["imresize", 84, 84],  # resize image vector to 84 x 84
    ["concat", 4, "append"],
    ["normalize"]
  ]
}
stack = build_preprocessing_stack(preprocessing_config)
config.state_shape = stack.shape(config.state_shape)
```

The first item in each list refers to the preprocessor to use, and the
remaining items are passed as \*args to the constructor. You can obtain
a list of valid preprocessor references from
[tensorforce.util.experiment\_util.py](https://github.com/reinforceio/tensorforce/blob/master/tensorforce/util/experiment_util.py).

Ready-to-use preprocessors
--------------------------

These are the preprocessors that come with TensorForce:

### Concat

Concatenate `concat_length` state vectors. Example: Used in Atari
problems to create the Markov property.

#### Parameters

```python
config = {
    'concat_length': int,  # how many states should be concatenated
    'dimension_position': string  # "prepend" or "append" - position where states should be concatenated
}
```

The `dimension_position` states in which dimension states are concatted.
For instance, let `concat_length = 2` and input `shape = (5, 7)`.

With `dimension_position = "prepend"` the output shape is `(2, 5, 7)`.

With `dimension_position = "append"`, the output shape is `(5, 7, 2)`.

### Grayscale

Turn a 3d image vector (HxWxC) into a 2d grayscale image (HxW).

#### Parameters

```python
config = {
    'weights': [int, int, int]  # list of channel weights (should sum to 1)
}
```

### Imresize

Resize a 2d image vector.

#### Parameters

```python
config = {
    'dimension_x': int,  # X (width) in px
    'dimension_y': int   # Y (height) in px
}
```

### Maximum

Return maximum of last `count` states.

#### Parameters

```python
config = {
    'count': int  # number of recent states to return maximum from
}
```

### Normalize

Normalize vector (feature scaling, interval 0-1).

#### Parameters

None

### Standardize

Standardize vector (normal distribution)

#### Parameters

None

Building your own preprocessor
------------------------------

All preprocessors should inherit from
`tensorforce.preprocessing.Preprocessor`.

For a start, take a look at the source of [Grayscale
preprocessor](https://github.com/reinforceio/tensorforce/blob/master/tensorforce/preprocessing/grayscale.py).

```python
from tensorforce.preprocessing import Preprocessor


class Grayscale(Preprocessor):
    # this is the default configuration
    default_config = {
        'weights': [0.299, 0.587, 0.114]
    }
    # this is the list of the *args to be parsed
    config_args = [
        'weights'
    ]

    # for instance, if the object is initialized with Grayscale([0.3, 0.4, 0.3]), then
    # self.weights = [0.3, 0.4, 0.3]

    def process(self, state):
        """
        Turn 3D color state into grayscale, thereby removing the last dimension.
        :param state: state input
        :return: new_state
        """
        # return processed state given the original state
        return (self.config.weights * state).sum(-1)

    def shape(self, original_shape):
        # return new shape given the original shape
        return list(original_shape[:-1])
```
