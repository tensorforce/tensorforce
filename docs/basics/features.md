Features
========


### Multi-input and non-sequential network architectures

Multi-input and other non-sequential networks are specified as list of layer lists, as opposed to
simply a list of layers for sequential networks. The following example illustrates how to specify
such a more complex network, by using the special layers `Register` and `Retrieve` to combine the
multiple sequential layer stacks.

```python
Agent.create(
    states=dict(
        observation=dict(type='float', shape=(16, 16, 3)),
        attributes=dict(type='int', shape=(4, 2), num_values=5)
    ),
    ...
    policy=[
        [
            dict(type='retrieve', tensors=['observation']),
            dict(type='conv2d', size=16),
            dict(type='flatten'),
            dict(type='register', tensor='obs-embedding')
        ],
        [
            dict(type='retrieve', tensors=['attributes']),
            dict(type='embedding', size=16),
            dict(type='flatten'),
            dict(type='register', tensor='attr-embedding')
        ],
        [
            dict(
                type='retrieve', tensors=['obs-embedding', 'attr-embedding'],
                aggregation='concat'
            ),
            dict(type='dense', size=32)
        ]
    ],
    ...
)
```



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



### Parallel environment execution

Execute multiple environments running locally in one call / batched:

```python
Runner(
    agent='benchmarks/configs/ppo1.json', environment='CartPole-v1',
    num_parallel=5
)
runner.run(num_episodes=100, batch_agent_calls=True)
```

Execute environments running in different processes whenever ready / unbatched:

```python
Runner(
    agent='benchmarks/configs/ppo1.json', environment='CartPole-v1',
    num_parallel=5, remote='multiprocessing'
)
runner.run(num_episodes=100)
```

Execute environments running on different machines, here using `run.py` instead
of `Runner`:

```bash
# Environment machine 1
python run.py --environment gym --level CartPole-v1 --remote socket-server \
    --port 65432

# Environment machine 2
python run.py --environment gym --level CartPole-v1 --remote socket-server \
    --port 65433

# Agent machine
python run.py --agent benchmarks/configs/ppo1.json --episodes 100 \
    --num-parallel 2 --remote socket-client --host 127.0.0.1,127.0.0.1 \
    --port 65432,65433 --batch-agent-calls
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
    num_iterations=100  # perform 100 update iterations on traces (more configurations possible)
)
```



### Save & restore

##### TensorFlow saver (full model)

```python
agent = Agent.create(...
    saver=dict(
        directory='data/checkpoints',
        frequency=100  # save checkpoint every 100 updates
    ), ...
)
...
agent.close()

# Restore latest agent checkpoint
agent = Agent.load(directory='data/checkpoints')
```


##### NumPy / HDF5 (only weights)

```python
agent = Agent.create(...)
...
agent.save(directory='data/checkpoints', format='numpy', append='episodes')

# Restore latest agent checkpoint
agent = Agent.load(directory='data/checkpoints', format='numpy')
```



### TensorBoard

```python
Agent.create(...
    summarizer=dict(
        directory='data/summaries',
        # list of labels, or 'all'
        labels=['entropy', 'graph', 'kl-divergence', 'loss', 'reward', 'update-norm']
    ), ...
)
```
