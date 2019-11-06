from tensorforce.environments import Environment

def resume_env(rank=1):
    return(Environment.create(environment='gym', level='CartPole-v1'))
