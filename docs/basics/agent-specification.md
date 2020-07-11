Agent specification
===================

Agents are instantiated via `Agent.create(agent=...)`, with either of the specification alternatives presented below (`agent` acts as `type` argument). It is recommended to pass as second argument `environment` the application `Environment` implementation, which automatically extracts the corresponding `states`, `actions` and `max_episode_timesteps` arguments of the agent.



### States and actions specification

A state/action value is specified as dictionary with mandatory attributes `type` (one of `'bool'`: binary, `'int'`: discrete, or `'float'`: continuous) and `shape` (a positive number or tuple thereof). Moreover, `'int'` values should additionally specify `num_values` (the fixed number of discrete options), whereas `'float'` values can specify bounds via `min/max_value`. If the state or action consists of multiple components, these are specified via an additional dictionary layer. The following example illustrates both possibilities:

```python
states = dict(
    observation=dict(type='float', shape=(16, 16, 3)),
    attributes=dict(type='int', shape=(4, 2), num_values=5)
)
actions = dict(type='float', shape=10)
```

Note: Ideally, the agent arguments `states` and `actions` are specified implicitly by passing the `environment` argument.



### How to specify modules

##### Dictionary with module type and arguments
```python
Agent.create(...
    policy=dict(network=dict(type='layered', layers=[dict(type='dense', size=32)])),
    memory=dict(type='replay', capacity=10000), ...
)
```


##### JSON specification file (plus additional arguments)
```python
Agent.create(...
    policy=dict(network='network.json'),
    memory=dict(type='memory.json', capacity=10000), ...
)
```


##### Module path (plus additional arguments)
```python
Agent.create(...
    policy=dict(network='my_module.TestNetwork'),
    memory=dict(type='tensorforce.core.memories.Replay', capacity=10000), ...
)
```


##### Callable or Type (plus additional arguments)
```python
Agent.create(...
    policy=dict(network=TestNetwork),
    memory=dict(type=Replay, capacity=10000), ...
)
```


##### Default module: only arguments or first argument
```python
Agent.create(...
    policy=dict(network=[dict(type='dense', size=32)]),
    memory=dict(capacity=10000), ...
)
```
