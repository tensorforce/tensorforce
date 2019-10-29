"""
Just a dummy env to let tensorforce be ready from the training side. This is only
a simple example, may need some adaptation.
"""

from tensorforce.environments import Environment
from simulation_base.parameters import S_DIM, A_DIM, amin, amax

class LBMENV(Environment):
    def __init__(self, rank=1):
        pass
    
    def reset(self):
        pass

    def execute(self, actions):
        pass
    

    def states(self):
        return dict(type='float',shape=(S_DIM,))

    def actions(self):
        return dict(type='float',shape=(A_DIM,), min_value = amin, max_value = amax)

    def close(self):
        pass

    def max_episode_timesteps(self):
        return None


def resume_env(rank=1):
    return(LBMENV(rank))
