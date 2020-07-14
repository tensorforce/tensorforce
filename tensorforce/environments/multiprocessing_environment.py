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

from multiprocessing import Pipe, Process

from tensorforce.environments import RemoteEnvironment


class MultiprocessingEnvironment(RemoteEnvironment):
    """
    An earlier version of this code (#634) was originally developed by Vincent Belus (@vbelus).
    """

    @classmethod
    def proxy_send(cls, connection, function, kwargs):
        connection[0].send(obj=(function, kwargs))

    @classmethod
    def proxy_receive(cls, connection):
        return connection[0].recv()

    @classmethod
    def proxy_close(cls, connection):
        connection[0].close()
        connection[1].join()

    @classmethod
    def remote_send(cls, connection, success, result):
        connection.send(obj=(success, result))

    @classmethod
    def remote_receive(cls, connection):
        return connection.recv()

    @classmethod
    def remote_close(cls, connection):
        connection.close()

    def __init__(self, environment, blocking=False, max_episode_timesteps=None, **kwargs):
        proxy_connection, remote_connection = Pipe(duplex=True)
        process = Process(
            target=self.__class__.remote, kwargs=dict(
                connection=remote_connection, environment=environment,
                max_episode_timesteps=max_episode_timesteps, **kwargs
            )
        )
        process.start()
        super().__init__(connection=(proxy_connection, process), blocking=blocking)
