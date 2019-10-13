Features
========


### Action masking
```python
agent = Agent.create(
    states=dict(type='float', shape=(10,)),
    actions=dict(type='int', shape=(), num_actions=3), ...
)
...
states = dict(
    state=np.random.random_sample(size=(10,)),  # regular state
    action_mask=[True, False, True]  # mask as '[ACTION-NAME]_mask'
)
action = agent.act(states=states)
assert action != 1
```


### Record & pretrain
```python
agent = Agent.create(...
    recorder=dict(
        directory='data/traces',
        frequency=100  # record a traces file every 100 episodes
    ), ...
)
...
agent.close()

# Pretrain agent on recorded traces
agent = Agent.create(...)
agent.pretrain(
    directory='data/traces',
    num_updates=100  # perform 100 updates on traces (other configurations possible)
)
```


### Save & restore
```python
agent = Agent.create(...
    saver=dict(
        directory='data/checkpoints',
        frequency=600  # save checkpoint every 600 seconds (10 minutes)
    ), ...
)
...
agent.close()

# Restore latest agent checkpoint
agent = Agent.load(directory='data/checkpoints')
```


### TensorBoard
```python
Agent.create(...
    summarizer=dict(
        directory='data/summaries',
        labels=['graph', 'losses', 'rewards'],  # list of labels, or 'all'
        frequency=100  # store values every 100 timesteps
        # (infrequent update summaries every update; other configurations possible)
    ), ...
)
```
