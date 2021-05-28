Features
========


### Multi-input and non-sequential network architectures

See [networks documentation](../modules.networks.html).



### Abort-terminal due to timestep limit

Besides `terminal=False` or `=0` for non-terminal and `terminal=True` or `=1` for true terminal, Tensorforce recognizes `terminal=2` as abort-terminal and handles it accordingly for reward estimation. Environments created via `Environment.create(..., max_episode_timesteps=?, ...)` will automatically return the appropriate terminal depending on whether an episode truly terminates or is aborted because it reached the time limit.



### Action masking

See also the [action-masking example](https://github.com/tensorforce/tensorforce/blob/master/examples/action_masking.py) for an environment implementation with built-in action masking.

```python
agent = Agent.create(
    states=dict(type='float', shape=(10,)),
    actions=dict(type='int', shape=(), num_values=3),
    ...
)
...
states = dict(
    state=np.random.random_sample(size=(10,)),  # state (default name: "state")
    action_mask=[True, False, True]  # mask as'[ACTION-NAME]_mask' (default name: "action")
)
action = agent.act(states=states)
assert action != 1
```



### Parallel environment execution

See also the [parallelization example](https://github.com/tensorforce/tensorforce/blob/master/examples/parallelization.py) for details on how to use this feature.

Execute multiple environments running locally in one call / batched:

```python
Runner(
    agent='benchmarks/configs/ppo1.json', environment='CartPole-v1',
    num_parallel=4
)
runner.run(num_episodes=100, batch_agent_calls=True)
```

Execute environments running in different processes whenever ready / unbatched:

```python
Runner(
    agent='benchmarks/configs/ppo1.json', environment='CartPole-v1',
    num_parallel=4, remote='multiprocessing'
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

See also the [save-load example](https://github.com/tensorforce/tensorforce/blob/master/examples/save_load_agent.py).


##### NumPy / HDF5 (only weights)

```python
agent = Agent.create(...)
...
agent.save(directory='data/checkpoints', format='numpy', append='episodes')

# Restore latest agent checkpoint
agent = Agent.load(directory='data/checkpoints', format='numpy')
```

See also the [save-load example](https://github.com/tensorforce/tensorforce/blob/master/examples/save_load_agent.py).


##### SavedModel export

See the [SavedModel example](https://github.com/tensorforce/tensorforce/blob/master/examples/export_saved_model.py) for details on how to use this feature.



### TensorBoard

```python
Agent.create(...
    summarizer=dict(
        directory='data/summaries',
        # list of labels, or 'all'
        labels=['entropy', 'kl-divergence', 'loss', 'reward', 'update-norm']
    ), ...
)
```



### Act-experience-update interaction

Instead of the default act-observe interaction pattern or the [Runner utility](../execution/runner.html), one can alternatively use the act-experience-update interface, which allows for more control over the experience the agent stores. See the [act-experience-update example](https://github.com/tensorforce/tensorforce/blob/master/examples/act_experience_update_interface.py) for details on how to use this feature. Note that a few stateful network layers will not be updated correctly in independent-mode (currently, `exponential_normalization`).



### Record & pretrain

See the [record-and-pretrain example](https://github.com/tensorforce/tensorforce/blob/master/examples/record_and_pretrain.py) for details on how to use this feature.
