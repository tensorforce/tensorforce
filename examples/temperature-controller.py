####
#### This is an example of a room that has a heater.  When the heater is off,
#### the room temperature approaches 0.0.  When the heater is on, the
#### temperature approaches 1.0.  The room temperature does not instantly go
#### to 0.0 or 1.0, but asymtotically approaches these temperatures with
#### an exponential decay term characterized by a time constant tau.
####
#### In this example, we will create the environment that simulates this,
#### and make a RL agent that will learn to turn the heater on and off to
### keep the temperature in a range of 0.4 - 0.6.
####
####


###-----------------------------------------------------------------------------
### Imports and setup
from tensorforce.environments import Environment
from tensorforce.agents import Agent
import numpy as np
import math



###-----------------------------------------------------------------------------
### Environment definition
class ThermostatEnvironment(Environment):
    """This class defines a simple thermostat environment.  It is a room with
    a heater, and when the heater is on, the room temperature will approach
    the max heater temperature (usually 1.0), and when off, the room will
    decay to a temperature of 0.0.  The exponential constant that determines
    how fast it approaches these temperatures over timesteps is tau.
    """
    def __init__(self):
        ## Some initializations.  Will eventually parameterize this in the constructor.
        # self.timestep = 0
        self.tau = 3.0
        #self.current_temp = 0.0
        # self.current_temp = np.array([np.random.random(size=(1,))])
        self.current_temp = np.random.random(size=(1,))

        super().__init__()


    def states(self):
        return dict(type='float', shape=(1,))


    def actions(self):
        """Action 0 means no heater, temperature approaches 0.0.  Action 1 means
        the heater is on and the room temperature approaches 1.0.
        """
        return dict(type='int', num_values=2)


    # Optional, should only be defined if environment has a natural maximum
    # episode length
    def max_episode_timesteps(self):
        # return 100
        return super().max_episode_timesteps()


    # Optional
    def close(self):
        super().close()


    def reset(self):
        """Reset state.
        """
        # state = np.random.random(size=(1,))
        self.timestep = 0
        self.current_temp = np.random.random(size=(1,))
        return self.current_temp


    def response(self, action):
        """Respond to an action.  When the action is 1, the temperature
        exponentially decays approaches 1.0.  When the action is 0,
        the current temperature decays towards 0.0.
        """
        # return np.array([action + (self.current_temp - action) * math.exp(-1.0 / self.tau)])
        return action + (self.current_temp - action) * math.exp(-1.0 / self.tau)


    def reward_compute(self):
        """ The reward here is 0 if the current temp is between 0.4 and 0.6,
        else it is distance the temp is away from the 0.4 or 0.6 boundary.
        """
        delta = abs(self.current_temp - 0.5)
        if delta < 0.1:
            return 0.0
        else:
            return -delta[0] + 0.1


    def execute(self, actions):
        ## Check the action is either 0 or 1 -- heater on or off.
        assert actions == 0 or actions == 1

        ## Increment timestamp
        self.timestep += 1
        
        ## Update the current_temp
        self.current_temp = self.response(actions)
        
        ## Compute the reward
        reward = self.reward_compute()

        ## The only way to go terminal is to exceed max_episode_timestamp.
        ## terminal == 0 means episode is not done
        ## terminal == 1 means it is done.
        terminal = False
        if self.timestep > self.max_episode_timesteps():
            terminal = True
        
        return self.current_temp, terminal, reward




###-----------------------------------------------------------------------------
### Create and initialize the environment
# environment = ThermostatEnvironment()
environment = environment = Environment.create(environment=ThermostatEnvironment, max_episode_timesteps=100)


## Try out one step, see results
# print(environment.execute(1))



###-----------------------------------------------------------------------------
### Setup the agent
agent = Agent.create(
    agent='tensorforce', environment=environment, update=64,
    objective='policy_gradient', reward_estimation=dict(horizon=1)
)



###-----------------------------------------------------------------------------
## Train the agent

## Environment needs to be reset to initialize it.
# environment.reset()


# Train for 200 episodes
for _ in range(200):
    states = environment.reset()
    terminal = False
    while not terminal:
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

# Evaluate for 100 episodes
sum_rewards = 0.0
for _ in range(100):
    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        actions, internals = agent.act(states=states, internals=internals, evaluation=True)
        states, terminal, reward = environment.execute(actions=actions)
        sum_rewards += reward

print('Mean episode reward:', sum_rewards / 100)

# Close agent and environment
agent.close()
environment.close()

