from tensorforce import Runner


runner = Runner(agent='benchmarks/configs/ppo1.json', environment='CartPole-v1')
runner.run(num_episodes=100)
runner.agent.save(directory='test/data', filename='ppo-checkpoint', format='checkpoint')
runner.agent.save(directory='test/data', filename='ppo-checkpoint', format='numpy')
runner.agent.save(directory='test/data', filename='ppo-checkpoint', format='hdf5')
# TODO: directory (since no filename)? doesn't work yet...
# runner.agent.save(directory='test/data', format='saved-model')
runner.close()
