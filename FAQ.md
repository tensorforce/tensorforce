TensorForce FAQ
===============

### 1. How can I use TensorForce in a new environment or application?

This depends on the control flow of your problem. For most simulations, it is convenient to
implement a binding to the TensorForce `Environment` class (various examples in contrib). The
advantage of this is that it allows you to use the existing execution scripts, in particular the
`Runner` utility. The general workflow is to copy one of the example scripts in the examples folder
which parse arguments and call the runner. The runner will then control execution by calling your
environment for the specified number of steps.

If you have a real-world environment, things are generally different as you may not be able to
delegate control flow to TensorForce. Instead, your external application might use TensorFlow as a
library, and call `act()` and `observe()` when new data is available. Consider the quickstart
example in the readme.


### 2. Why is my algorithm not learning?

Generally, this is either because there is a bug in our implementation or a problem in your
configuration or application. The reality of reinforcement learning is that getting things to work
is *very* difficult and we will not be able to tell you why your specific thing is not working.
Newcomers with some experience in deep learning, where successfully training small networks is
easy, often carry over this expectation to reinforcement learning. Issues that simply
ask for configuration help without doing some research (small ablation analysis, using some
known hyper-parameters from papers, or reasonable argument why something should work) will be
closed with reference to this document.

Please appreciate that for almost every problem, none of the default configurations will work,
usually because batch sizes and learning rates are wrong, or you need vastly more data. Substantial
practical experience is required to get an intuition for what is possible with which amount of data
for a given problem. Reproducing papers is extremely difficult. From a user perspective, the
expectation should be that to get an algorithm to work on *any* problem, hyper-parameter tuning
is required.

That being said, there are small implementation issues in some of the Q-models due to the move to
full TensorFlow code, and the current recommendation is to use PPO unless there is a good reason
not to.


### 3. Can you implement paper X?

We get many feature requests and the answer to most is "maybe". Reinforcement learning is a very
active, fast moving field, but at the same time very immature as a technology. This means most new
approaches will likely not be practically relevant going forward, this is the nature of research,
even though these approaches inform the development of more mature algorithms later.

Further, new approaches are often unnecessarily complex, which often only becomes clear in
hindsight. For example, PPO both performs much better than TRPO and is much simpler to implement.
TensorForce is not meant to be a collection of every new available trick. This is in particular not
possible due to the architecture choices we have made to design full TensorFlow reinforcement
learning graphs. Integrating new techniques into this architecture tends to require *much* higher
effort than just implementing the new method from scratch in a separate script.

For this reason, we mostly do not implement new papers straight away unless they are extremely
convincing and have a good implementation difficulty to return ratio.


### 4. How can I use an evolutionary or natural gradient optimizer?

By changing the type of the optimizer where appropriate. For example, a vanilla policy gradient may
use an evolutionary optimizer via:

```python
optimizer=dict(
    type='evolutionary',
    learning_rate=1e-2
)
```

and a natural gradient optimizer via:

```python
optimizer=dict(
    type='natual_gradient',
    learning_rate=1e-2
)
```

Please note that not every model can sensibly make use of every optimizer. Read up on the
individual model, for example, TRPO is by default a natural gradient model.


### 5. What is deterministic mode? How does it relate to evaluation?

The deterministic flag on ```act``` only concerns the action selection. It does
not affect whether training is performed or not. Training is controlled via
```observe``` calls. The deterministic flag is relevant with regard to stochastic
and deterministic policies. For example, policy gradient models typically
assume a stochastic policy during training (unless when using deterministic policy
gradients). Q-models, which we have also implemented inheriting from a distribution
model (see our blog posts on architecture), deterministically sample their action
via their greedy maximisation.

When evaluating the final trained model, one would typically act deterministically to
avoid sampling random actions.

### 6. How do I specify multiple actions?

Actions can either be specified as single actions which just require a dict specifying
a ```type``` and ```shape```, or a ```num_actions``` parameter in the case of discrete actions. 
A ```shape``` can be used to specify multiple actions of the same type, e.g.
```actions=dict(type='float', shape=(5,), min=-2, max=2)``` specifies 5 float actions
with the same range. Please note that using the min/max parameter results in
a Beta distribution instead of a Gaussian distribution being used for continuous 
actions. If a single action is desired, use ```shape=()```.

The parameter ```num_actions``` is only used for discrete
actions and describes the number of different options for a single discrete action. That is,
```dict(type='int', shape=(2,), num_actions=4)``` means the agent will output two actions
per step, each giving values in [0, 1, 2, 3].

There are no restrictions on combining multiple action types. Available action types
are ```float,int,bool```. In the case of multiple actions, the agent expects a dict containing
action names and their types. For example,

```python
actions=dict(
   discrete_action=dict(type='int', num_actions=15),
   continous_action=dict(type='float', shape=()),
)
```  

will lead to the agent outputting the following action dict (example action):

```python
actions = agent.act(states=state)

>>> print actions
>>> dict(discrete_action=3, continuous_action=0.332)
``` 

In the case of multiple actions, the values will be arrays. It is a good idea
to use expressive action names within your application to keep track of 
each action and its purpose.