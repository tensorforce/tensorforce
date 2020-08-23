# Tensorforce Benchmarks

Use the `run.py` script with the following arguments to produce benchmarks:

```bash
python run.py --agent benchmarks/configs/ppo.json --environment gym --level CartPole-v1 \
    --episodes 100 --repeat 10 --path benchmarks/gym-cartpole/ppo
```

To run a full benchmark of a config in the `configs` subfolder, call the `benchmark.sh` bash script with the config name:

```bash
benchmarks/benchmark.sh ppo
```
