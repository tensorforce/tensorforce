import os

from tensorforce import Runner


os.remove('test/data/checkpoint')
os.remove('test/data/ppo-checkpoint-1.data-00000-of-00001')
os.remove('test/data/ppo-checkpoint-1.index')
os.remove('test/data/ppo-checkpoint.json')
os.remove('test/data/ppo-checkpoint.npz')
os.remove('test/data/ppo-checkpoint.hdf5')

os.rmdir('test/data/ppo-checkpoint/assets')
os.remove('test/data/ppo-checkpoint/variables/variables.data-00000-of-00001')
os.remove('test/data/ppo-checkpoint/variables/variables.index')
os.rmdir('test/data/ppo-checkpoint/variables')
os.remove('test/data/ppo-checkpoint/saved_model.pb')
os.rmdir('test/data/ppo-checkpoint')


runner = Runner(
    agent=dict(
        agent='benchmarks/configs/ppo.json',
        config=dict(device='CPU'),
        recorder=dict(directory='test/data/ppo-traces', start=80)
    ), environment='benchmarks/configs/cartpole.json'
)
runner.run(num_episodes=100)
runner.agent.save(directory='test/data', filename='ppo-checkpoint', format='checkpoint')
runner.agent.save(directory='test/data', filename='ppo-checkpoint', format='numpy')
runner.agent.save(directory='test/data', filename='ppo-checkpoint', format='hdf5')
runner.agent.save(directory='test/data', filename='ppo-checkpoint', format='saved-model')
runner.close()
