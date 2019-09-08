# Copyright 2018 Tensorforce Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os
import pickle

import ConfigSpace as cs
from hpbandster.core.nameserver import NameServer, nic_name_to_host
from hpbandster.core.result import json_result_logger, logged_results_to_HBS_result
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
import numpy as np

from tensorforce.environments import Environment
from tensorforce.execution import Runner


class TensorforceWorker(Worker):

    def __init__(self, *args, environment=None, **kwargs):
        # def __init__(self, run_id, nameserver=None, nameserver_port=None, logger=None, host=None, id=None, timeout=None):
        super().__init__(*args, **kwargs)
        assert environment is not None
        self.environment = environment

    def compute(self, config_id, config, budget, working_directory):
        if self.environment.max_episode_timesteps() is None:
            min_capacity = 1000 + config['batch_size']
        else:
            min_capacity = self.environment.max_episode_timesteps() + config['batch_size']
        max_capacity = 100000
        capacity = min(max_capacity, max(min_capacity, config['memory'] * config['batch_size']))
        frequency = max(16, int(config['frequency'] * config['batch_size']))

        if config['ratio_based'] == 'yes':
            ratio_based = True
            clipping_value = config['clipping_value']
        else:
            ratio_based = False
            clipping_value = 0.0

        if config['baseline'] == 'no':
            baseline_policy = None
            baseline_network = None
            baseline_objective = None
            baseline_optimizer = None
            estimate_horizon = False
            estimate_terminal = False
            estimate_advantage = False
        else:
            estimate_horizon = 'early'
            estimate_terminal = True
            estimate_advantage = (config['estimate_advantage'] == 'yes')
            if config['baseline'] == 'same-policy-noopt':
                baseline_policy = 'same'
                baseline_network = None
                baseline_objective = None
                baseline_optimizer = None
            else:
                baseline_objective = dict(type='state_value', huber_loss=0.0, mean_over_actions=False)
                baseline_optimizer = dict(type='adam', learning_rate=config['baseline_learning_rate'])
                if config['baseline'] == 'auto':
                    baseline_policy = None
                    baseline_network = dict(type='auto', internal_rnn=False)
                elif config['baseline'] == 'same-network':
                    baseline_policy = None
                    baseline_network = 'same'
                elif config['baseline'] == 'same-policy':
                    baseline_policy = 'same'
                    baseline_network = None
                else:
                    assert False

        if config['entropy_regularization'] < 3e-5:  # yes/no better
            entropy_regularization = 0.0
        else:
            entropy_regularization = config['entropy_regularization']

        agent = dict(
            agent='policy',
            network=dict(type='auto', internal_rnn=False),
            memory=dict(type='replay', capacity=capacity),
            update=dict(unit='timesteps', batch_size=config['batch_size'], frequency=frequency),
            optimizer=dict(type='adam', learning_rate=config['learning_rate']),
            objective=dict(
                type='policy_gradient', ratio_based=ratio_based, clipping_value=clipping_value,
                mean_over_actions=False
            ),
            reward_estimation=dict(
                horizon=config['horizon'], discount=config['discount'],
                estimate_horizon=estimate_horizon, estimate_actions=False,
                estimate_terminal=estimate_terminal, estimate_advantage=estimate_advantage
            ),
            baseline_policy=baseline_policy, baseline_network=baseline_network,
            baseline_objective=baseline_objective, baseline_optimizer=baseline_optimizer,
            preprocessing=None,
            l2_regularization=0.0, entropy_regularization=entropy_regularization
        )

        # num_episodes = list()
        final_reward = list()
        max_reward = list()
        rewards = list()

        for n in range(round(budget)):
            runner = Runner(agent=agent, environment=self.environment)

            # performance_threshold = runner.environment.max_episode_timesteps() - agent['reward_estimation']['horizon']

            # def callback(r):
            #     return True

            runner.run(num_episodes=500, use_tqdm=False)
            runner.close()

            # num_episodes.append(len(runner.episode_rewards))
            final_reward.append(float(np.mean(runner.episode_rewards[-20:], axis=0)))
            average_rewards = [
                float(np.mean(runner.episode_rewards[n: n + 20], axis=0))
                for n in range(len(runner.episode_rewards) - 20)
            ]
            max_reward.append(float(np.amax(average_rewards, axis=0)))
            rewards.append(list(runner.episode_rewards))

        # mean_num_episodes = float(np.mean(num_episodes, axis=0))
        mean_final_reward = float(np.mean(final_reward, axis=0))
        mean_max_reward = float(np.mean(max_reward, axis=0))
        # loss = mean_num_episodes - mean_final_reward - mean_max_reward
        loss = -mean_final_reward - mean_max_reward

        return dict(loss=loss, info=dict(rewards=rewards))

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        configspace = cs.ConfigurationSpace()

        memory = cs.hyperparameters.UniformIntegerHyperparameter(name='memory', lower=2, upper=25)
        configspace.add_hyperparameter(hyperparameter=memory)

        batch_size = cs.hyperparameters.UniformIntegerHyperparameter(
            name='batch_size', lower=32, upper=8192, log=True
        )
        configspace.add_hyperparameter(hyperparameter=batch_size)

        frequency = cs.hyperparameters.UniformFloatHyperparameter(
            name='frequency', lower=3e-2, upper=1.0, log=True
        )
        configspace.add_hyperparameter(hyperparameter=frequency)

        learning_rate = cs.hyperparameters.UniformFloatHyperparameter(
            name='learning_rate', lower=1e-5, upper=3e-2, log=True
        )
        configspace.add_hyperparameter(hyperparameter=learning_rate)

        horizon = cs.hyperparameters.UniformIntegerHyperparameter(
            name='horizon', lower=1, upper=50
        )
        configspace.add_hyperparameter(hyperparameter=horizon)

        discount = cs.hyperparameters.UniformFloatHyperparameter(
            name='discount', lower=0.8, upper=1.0, log=True
        )
        configspace.add_hyperparameter(hyperparameter=discount)

        ratio_based = cs.hyperparameters.CategoricalHyperparameter(
            name='ratio_based', choices=('no', 'yes')
        )
        configspace.add_hyperparameter(hyperparameter=ratio_based)

        clipping_value = cs.hyperparameters.UniformFloatHyperparameter(
            name='clipping_value', lower=0.05, upper=0.5
        )
        configspace.add_hyperparameter(hyperparameter=clipping_value)

        baseline = cs.hyperparameters.CategoricalHyperparameter(
            name='baseline',
            choices=('no', 'auto', 'same-network', 'same-policy', 'same-policy-noopt')
        )
        configspace.add_hyperparameter(hyperparameter=baseline)

        baseline_learning_rate = cs.hyperparameters.UniformFloatHyperparameter(
            name='baseline_learning_rate', lower=1e-5, upper=3e-2, log=True
        )
        configspace.add_hyperparameter(hyperparameter=baseline_learning_rate)

        estimate_advantage = cs.hyperparameters.CategoricalHyperparameter(
            name='estimate_advantage', choices=('no', 'yes')
        )
        configspace.add_hyperparameter(hyperparameter=estimate_advantage)

        entropy_regularization = cs.hyperparameters.UniformFloatHyperparameter(
            name='entropy_regularization', lower=1e-5, upper=1.0, log=True
        )
        configspace.add_hyperparameter(hyperparameter=entropy_regularization)

        configspace.add_condition(
            condition=cs.EqualsCondition(child=clipping_value, parent=ratio_based, value='yes')
        )

        configspace.add_condition(
            condition=cs.NotEqualsCondition(
                child=baseline_learning_rate, parent=baseline, value='no'
            )
        )

        configspace.add_condition(
            condition=cs.NotEqualsCondition(
                child=estimate_advantage, parent=baseline, value='no'
            )
        )

        return configspace


def main():
    parser = argparse.ArgumentParser(description='Tensorforce hyperparameter tuner')
    parser.add_argument(
        'environment', help='Environment (name, configuration JSON file, or library module)'
    )
    parser.add_argument(
        '-l', '--level', type=str, default=None,
        help='Level or game id, like `CartPole-v1`, if supported'
    )
    parser.add_argument(
        '-m', '--max-repeats', type=int, default=1, help='Maximum number of repetitions'
    )
    parser.add_argument(
        '-n', '--num-iterations', type=int, default=1, help='Number of BOHB iterations'
    )
    parser.add_argument(
        '-d', '--directory', type=str, default='tuner', help='Output directory'
    )
    parser.add_argument(
        '-r', '--restore', type=str, default=None, help='Restore from given directory'
    )
    parser.add_argument('--id', type=str, default='worker', help='Unique worker id')
    args = parser.parse_args()

    if False:
        host = nic_name_to_host(nic_name=None)
        port = 123
    else:
        host = 'localhost'
        port = None

    server = NameServer(run_id=args.id, working_directory=args.directory, host=host, port=port)
    nameserver, nameserver_port = server.start()

    if args.level is None:
        environment = Environment.create(environment=args.environment)
    else:
        environment = Environment.create(environment=args.environment, level=args.level)

    worker = TensorforceWorker(
        environment=environment, run_id=args.id, nameserver=nameserver,
        nameserver_port=nameserver_port, host=host
    )
    # TensorforceWorker(run_id, nameserver=None, nameserver_port=None, logger=None, host=None, id=None, timeout=None)
    # logger: logging.logger instance, logger used for debugging output
    # id: anything with a __str__method, if multiple workers are started in the same process, you MUST provide a unique id for each one of them using the `id` argument.
    # timeout: int or float, specifies the timeout a worker will wait for a new after finishing a computation before shutting down. Towards the end of a long run with multiple workers, this helps to shutdown idling workers. We recommend a timeout that is roughly half the time it would take for the second largest budget to finish. The default (None) means that the worker will wait indefinitely and never shutdown on its own.

    worker.run(background=True)

    # config = cs.sample_configuration().get_dictionary()
    # print(config)
    # res = worker.compute(config=config, budget=1, working_directory='.')
    # print(res)

    if args.restore is None:
        previous_result = None
    else:
        previous_result = logged_results_to_HBS_result(directory=args.restore)

    result_logger = json_result_logger(directory=args.directory, overwrite=True)  # ???

    optimizer = BOHB(
        configspace=worker.get_configspace(), min_budget=0.5, max_budget=float(args.max_repeats),
        run_id=args.id, working_directory=args.directory,
        nameserver=nameserver, nameserver_port=nameserver_port, host=host,
        result_logger=result_logger, previous_result=previous_result
    )
    # BOHB(configspace=None, eta=3, min_budget=0.01, max_budget=1, min_points_in_model=None, top_n_percent=15, num_samples=64, random_fraction=1 / 3, bandwidth_factor=3, min_bandwidth=1e-3, **kwargs)
    # Master(run_id, config_generator, working_directory='.', ping_interval=60, nameserver='127.0.0.1', nameserver_port=None, host=None, shutdown_workers=True, job_queue_sizes=(-1,0), dynamic_queue_size=True, logger=None, result_logger=None, previous_result = None)
    # logger: logging.logger like object, the logger to output some (more or less meaningful) information

    results = optimizer.run(n_iterations=args.num_iterations)
    # optimizer.run(n_iterations=1, min_n_workers=1, iteration_kwargs={})
    # min_n_workers: int, minimum number of workers before starting the run

    optimizer.shutdown(shutdown_workers=True)
    server.shutdown()

    with open(os.path.join(args.directory, 'results.pkl'), 'wb') as filehandle:
        pickle.dump(results, filehandle)

    print('Best found configuration:', results.get_id2config_mapping()[results.get_incumbent_id()]['config'])
    print('Runs:', results.get_runs_by_id(config_id=results.get_incumbent_id()))
    print('A total of {} unique configurations where sampled.'.format(len(results.get_id2config_mapping())))
    print('A total of {} runs where executed.'.format(len(results.get_all_runs())))
    print('Total budget corresponds to {:.1f} full function evaluations.'.format(
        sum([r.budget for r in results.get_all_runs()]) / args.max_repeats)
    )


if __name__ == '__main__':
    main()
