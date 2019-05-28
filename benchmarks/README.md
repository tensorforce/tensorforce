# Tensorforce Benchmarks

Use the `run.py` script with the following arguments to produce benchmarks:

```bash
python run.py benchmarks/configs/ppo.json gym -l CartPole-v1 -e 300 -r 10 -p benchmarks/gym-cartpole/ppo
```

To run a full benchmark of a config in the `configs` subfolder, call the `benchmark.sh` bash script with the config name:

```bash
benchmarks/benchmark.sh ppo
```
