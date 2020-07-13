Features
========


### Multi-input and non-sequential network architectures

See [networks documentation](../modules.networks.html).



### Abort-terminal due to timestep limit

Besides `terminal=False` or `=0` for non-terminal and `terminal=True` or `=1` for true terminal, Tensorforce recognizes `terminal=2` as abort-terminal and handles it accordingly for reward estimation. Environments created via `Environment.create(..., max_episode_timesteps=?, ...)` will automatically return the appropriate terminal depending on whether an episode truly terminates or is aborted because it reached the time limit.



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



### Record & pretrain

See [record-and-pretrain example](https://github.com/tensorforce/tensorforce/blob/master/examples/record_and_pretrain.py).
