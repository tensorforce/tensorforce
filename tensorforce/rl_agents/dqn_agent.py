from tensorforce.replay_memories.replay_memory import ReplayMemory
from tensorforce.rl_agents.rl_agent import RLAgent
from tensorforce.value_functions.deep_q_network import DeepQNetwork

"""
Standard DQN. The piece de resistance of deep reinforcement learning.
Chooses from one of a number of discrete actions by taking the maximum Q-value
from the value function with one output neuron per available action.
"""


class DQNAgent(RLAgent):

    def __init__(self, agent_config, value_config):
        """
        Initialize a vanilla DQN agent as described in
        http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html.

        :param agent_config: Configuration parameters for agent
        :param value_config: Configuration parameters for deep Q network,
        i.e. network configuration
        """
        self.agent_config = agent_config
        self.value_function = DeepQNetwork(value_config)

        self.memory = ReplayMemory(agent_config['capacity'],
                                   agent_config['state_shape'],
                                   agent_config['state_type'],
                                   agent_config['action_shape'],
                                   agent_config['action_type'],
                                   agent_config['reward_type'],
                                   agent_config['concat'],
                                   agent_config['concat_lenght'],
                                   agent_config['deterministic_mode'])
        self.step_count = 0
        self.batch_size = agent_config['batch_size']
        self.update_rate = agent_config['update_rate']
        self.min_replay_size = agent_config['min_replay_size']

    def execute_step(self, state, reward, is_terminal):
        """
        Executes one reinforcement learning step. Implicitly computes updates
        according to the update frequency.

        :param state: Observed state tensor.
        :param reward: Observed reward.
        :param is_terminal: Indicates whether state terminates episode.
        :return: Which action to take
        """
        action = self.value_function.evaluate(state)

        if self.step_count > self.min_replay_size and self.step_count % self.update_rate == 0:
            self.value_function.update(self.memory.sample_batch(self.batch_size))

        self.step_count += 1

        return action

    def evaluate_state(self, state):
        """
        Evaluates the value function without learning, e.g. to use a trained model.

        :param state: A state tensor
        :return: An action
        """

        return self.value_function.evaluate(state)
