"""
Just a dummy env to let tensorforce be ready from the training side. This is only
a simple example, may need some adaptation.
"""

from tensorforce.environments import Environment
environment_example = Environment.create(environment='gym', level='CartPole-v1')

class dummy_env(Environment):
    def __init__(self, rank=1):
        pass
    
    def reset(self):
        pass

    def execute(self, actions):
        pass
    

    def states(self):
        return environment_example.states()

    def actions(self):
        return environment_example.actions()

    def close(self):
        pass

    def max_episode_timesteps(self):
        return environment_example.max_episode_timesteps()


def resume_env(rank=1):
    return(dummy_env(rank))
