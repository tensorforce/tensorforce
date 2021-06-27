# Copyright 2020 Tensorforce Team. All Rights Reserved.
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

import numpy as np

from tensorforce.environments import Environment


class CartPole(Environment):
    """
    Based on OpenAI Gym version
    (https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py)
    """

    def __init__(
        self,
        # Physics parameters
        pole_mass=(0.05, 0.5),  # 0.1
        pole_length=(0.1, 1.0),  # 0.5, actually half the pole's length
        cart_mass=(0.5, 1.5),
        relative_force=(0.75, 1.5),  # 1.0, relative to gravity
        gravity=9.8,
        # State space
        state_velocities=True,
        state_location=False,  # true
        state_initial_max_angle=0.05,
        state_initial_max_angle_velocity=0.05,
        state_initial_max_velocity=0.05,
        # Action space
        action_timedelta=0.02,
        action_continuous=False,
        action_noop=True  # false
    ):
        super().__init__()

        # Physics parameters
        if isinstance(pole_mass, tuple):
            assert len(pole_mass) == 2 and 0.0 < pole_mass[0] < pole_mass[1]
            self._pole_mass_range = (float(pole_mass[0]), float(pole_mass[1]))
        else:
            assert pole_mass > 0.0
            self._pole_mass_range = (float(pole_mass), float(pole_mass))
        if isinstance(pole_length, tuple):
            assert len(pole_length) == 2 and 0.0 < pole_length[0] < pole_length[1]
            self._pole_length_range = (float(pole_length[0]), float(pole_length[1]))
        else:
            assert pole_length > 0.0
            self._pole_length_range = (float(pole_length), float(pole_length))
        if isinstance(cart_mass, tuple):
            assert len(cart_mass) == 2 and 0.0 < cart_mass[0] < cart_mass[1]
            self._cart_mass_range = (float(cart_mass[0]), float(cart_mass[1]))
        else:
            assert cart_mass > 0.0
            self._cart_mass_range = (float(cart_mass), float(cart_mass))
        if isinstance(relative_force, tuple):
            assert len(relative_force) == 2 and 0.0 < relative_force[0] < relative_force[1]
            self._relative_force_range = (float(relative_force[0]), float(relative_force[1]))
        else:
            assert relative_force > 0.0
            self._relative_force_range = (float(relative_force), float(relative_force))
        assert gravity > 0.0
        self._gravity = float(gravity)

        # State space
        state_indices = [2]
        self._state_velocities = bool(state_velocities)
        if self._state_velocities:
            state_indices.append(3)
            state_indices.append(1)
        self._state_location = bool(state_location)
        if self._state_location:
            state_indices.append(0)
        self._state_indices = np.array(state_indices, np.int32)
        self._state_initials = np.array([[
            0.0, float(state_initial_max_velocity),
            float(state_initial_max_angle), float(state_initial_max_angle_velocity)
        ]], dtype=np.float32)

        # Action space
        self._action_timedelta = float(action_timedelta)  # in seconds
        assert not action_continuous or action_noop
        self._action_continuous = bool(action_continuous)
        self._action_noop = bool(action_noop)

        # State bounds
        angle_bound = float(np.pi) / 4.0
        max_angle_acc_in_zero = self._relative_force_range[1] * self._gravity / \
            (self._cart_mass_range[0] + self._pole_mass_range[0]) / \
            self._pole_length_range[0] / \
            (4.0 / 3.0 - self._pole_mass_range[1] / (self._cart_mass_range[0] + self._pole_mass_range[0]))
        min_angle_acc_in_zero = self._relative_force_range[0] * self._gravity / \
            (self._cart_mass_range[1] + self._pole_mass_range[1]) / \
            self._pole_length_range[1] / \
            (4.0 / 3.0 - self._pole_mass_range[0] / (self._cart_mass_range[1] + self._pole_mass_range[1]))
        max_loc_acc_in_zero = (self._relative_force_range[1] * self._gravity - \
            self._pole_mass_range[0] * self._pole_length_range[0] * min_angle_acc_in_zero) / \
            (self._cart_mass_range[0] + self._pole_mass_range[0])
        angle_vel_bound = max_angle_acc_in_zero * self._action_timedelta * 10.0
        loc_vel_bound = max_loc_acc_in_zero * self._action_timedelta * 10.0
        if self._state_location:
            loc_bound = loc_vel_bound
        else:
            loc_bound = np.inf
        self._state_bounds = np.array(
            [[loc_bound, loc_vel_bound, angle_bound, angle_vel_bound]], dtype=np.float32
        )
        assert (self._state_bounds > 0.0).all()

    def states(self):
        return dict(
            type='float', shape=tuple(self._state_indices.shape),
            min_value=-self._state_bounds[0, self._state_indices],
            max_value=self._state_bounds[0, self._state_indices]
        )

    def actions(self):
        if self._action_continuous:
            return dict(type='float', shape=())
        elif self._action_noop:
            return dict(type='int', shape=(), num_values=3)
        else:
            return dict(type='int', shape=(), num_values=2)

    def is_vectorizable(self):
        return True

    def reset(self, num_parallel=None):
        self._is_parallel = (num_parallel is not None)
        if self._is_parallel:
            self._parallel_indices = np.arange(num_parallel)
        else:
            self._parallel_indices = np.arange(1)

        # Physics parameters
        self._pole_mass = float(np.random.uniform(low=self._pole_mass_range[0], high=self._pole_mass_range[1]))
        self._pole_length = float(np.random.uniform(low=self._pole_length_range[0], high=self._pole_length_range[1]))
        self._cart_mass = float(np.random.uniform(low=self._cart_mass_range[0], high=self._cart_mass_range[1]))
        self._relative_force = float(np.random.uniform(low=self._relative_force_range[0], high=self._relative_force_range[1]))

        # Initialize state
        initials = np.tile(self._state_initials, reps=(self._parallel_indices.shape[0], 1))
        self._states = np.random.uniform(low=-initials, high=initials)

        if self._is_parallel:
            return self._parallel_indices.copy(), self._states[:, self._state_indices]
        else:
            return self._states[0, self._state_indices]

    def execute(self, actions):
        if not self._is_parallel:
            actions = np.expand_dims(actions, axis=0)

        # Split state into components
        loc = self._states[:, 0]
        loc_vel = self._states[:, 1]
        angle = self._states[:, 2]
        angle_vel = self._states[:, 3]

        # Make action continuous
        if self._action_continuous:
            force = actions
        else:
            force = np.where(actions == 2, 0.0, np.where(actions == 1, 1.0, -1.0))
        force *= self._relative_force * self._gravity

        # Compute accelerations (https://coneural.org/florian/papers/05_cart_pole.pdf)
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        total_mass = self._cart_mass + self._pole_mass
        pole_mass_length = self._pole_mass * self._pole_length
        bracket = (force + pole_mass_length * angle_vel * angle_vel * sin_angle) / total_mass
        denom = self._pole_length * (4.0 / 3.0 - (self._pole_mass * cos_angle * cos_angle) / total_mass)
        angle_acc = (self._gravity * sin_angle - cos_angle * bracket) / denom
        loc_acc = bracket - pole_mass_length * angle_acc * cos_angle / total_mass

        # Integration
        deriv = np.stack([loc_vel, loc_acc, angle_vel, angle_acc], axis=1)
        self._states += self._action_timedelta * deriv

        # Terminal
        terminal = (np.abs(self._states) > self._state_bounds).any(axis=1)

        # Reward
        reward = np.ones_like(terminal, dtype=np.float32)

        if self._is_parallel:
            self._parallel_indices = self._parallel_indices[~terminal]
            self._states = self._states[~terminal]
            return self._parallel_indices.copy(), self._states[:, self._state_indices], terminal, reward
        else:
            return self._states[0, self._state_indices], terminal.item(), reward.item()
