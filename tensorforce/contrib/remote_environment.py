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


from tensorforce.environments import Environment
import socket
import msgpack
import msgpack_numpy as mnp
import errno
import os
from tensorforce import TensorForceError
import logging
import time


class RemoteEnvironment(Environment):
    def __init__(self, host="localhost", port=6025):
        """
        A remote Environment that one can connect to through tcp.
        Implements a simple msgpack protocol to get the step/reset/etc.. commands to the
        remote server and simply waits (blocks) for a response.

        Args:
                host (str): The hostname to connect to.
                port (int): The port to connect to.
        """
        Environment.__init__(self)
        self.port = int(port) or 6025
        self.host = host or "localhost"
        self.socket = None
        # The size of the response buffer (depends on the Env's observation-space).
        self.buffer_size = 8192

        # Cache the last received observation (through socket) here.
        self.last_observation = None

    def __str__(self):
        return "RemoteEnvironment({}:{}{})".format(self.host, self.port, " [connected]" if self.socket else "")

    def close(self):
        """
        Same as disconnect method.
        """
        self.disconnect()

    def connect(self, timeout=600):
        """
        Starts the server tcp connection on the given host:port.

        Args:
            timeout (int): The time (in seconds) for which we will attempt a connection to the remote
                (every 5sec). After that (or if timeout is None or 0), an error is raised.
        """
        # If we are already connected, return error.
        if self.socket:
            raise TensorForceError("Already connected to {}:{}. Only one connection allowed at a time. " +
                                   "Close first by calling `close`!".format(self.host, self.port))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        if timeout < 5 or timeout is None:
            timeout = 5

        err = 0
        start_time = time.time()
        while time.time() - start_time < timeout:
            self.socket.settimeout(5)
            err = self.socket.connect_ex((self.host, self.port))
            if err == 0:
                break
            time.sleep(1)
        if err != 0:
            raise TensorForceError("Error when trying to connect to {}:{}: errno={} errcode='{}' '{}'".
                                   format(self.host, self.port, err, errno.errorcode[err], os.strerror(err)))

    def disconnect(self):
        """
        Ends our server tcp connection.
        """
        # If we are not connected, return error.
        if not self.socket:
            logging.warning("No active socket to close!")
            return
        # Close our socket.
        self.socket.close()
        self.socket = None

    @property
    def current_state(self):
        return self.last_observation


class MsgPackNumpyProtocol(object):
    """
    A simple protocol to communicate over tcp sockets, which can be used by RemoteEnvironment implementations.
    The protocol is based on msgpack-numpy encoding and decoding.

    Each message has a simple 8-byte header, which encodes the length of the subsequent msgpack-numpy
    encoded byte-string.
    All messages received need to have the 'status' field set to 'ok'. If 'status' is set to 'error',
    the field 'message' should be populated with some error information.

    Examples:
    client sends: "[8-byte header]msgpack-encoded({"cmd": "seed", "value": 200})"
    server responds: "[8-byte header]msgpack-encoded({"status": "ok", "value": 200})"

    client sends: "[8-byte header]msgpack-encoded({"cmd": "reset"})"
    server responds: "[8-byte header]msgpack-encoded({"status": "ok"})"

    client sends: "[8-byte header]msgpack-encoded({"cmd": "step", "action": 5})"
    server responds: "[8-byte header]msgpack-encoded({"status": "ok", "obs_dict": {... some observations},
    "reward": -10.0, "is_terminal": False})"
    """
    def __init__(self, max_msg_len=8192):
        """
        Args:
            max_msg_len (int): The maximum number of bytes to read from the socket.
        """
        self.max_msg_len = max_msg_len
        # Make all msgpack methods use the numpy-aware de/encoders.
        mnp.patch()

    def send(self, message, socket_):
        """
        Sends a message (dict) to the socket. Message consists of a 8-byte len header followed by a msgpack-numpy
            encoded dict.

        Args:
            message: The message dict (e.g. {"cmd": "reset"})
            socket_: The python socket object to use.
        """
        if not socket_:
            raise TensorForceError("No socket given in call to `send`!")
        elif not isinstance(message, dict):
            raise TensorForceError("Message to be sent must be a dict!")
        message = msgpack.packb(message)
        len_ = len(message)
        # prepend 8-byte len field to all our messages
        socket_.send(bytes("{:08d}".format(len_), encoding="ascii") + message)

    def recv(self, socket_, encoding=None):
        """
        Receives a message as msgpack-numpy encoded byte-string from the given socket object.
        Blocks until something was received.

        Args:
            socket_: The python socket object to use.
            encoding (str): The encoding to use for unpacking messages from the socket.
        Returns: The decoded (as dict) message received.
        """
        unpacker = msgpack.Unpacker(encoding=encoding)

        # Wait for an immediate response.
        response = socket_.recv(8)  # get the length of the message
        if response == b"":
            raise TensorForceError("No data received by socket.recv in call to method `recv` " +
                                   "(listener possibly closed)!")
        orig_len = int(response)
        received_len = 0
        while True:
            data = socket_.recv(min(orig_len - received_len, self.max_msg_len))
            # There must be a response.
            if not data:
                raise TensorForceError("No data of len {} received by socket.recv in call to method `recv`!".
                                       format(orig_len - received_len))
            data_len = len(data)
            received_len += data_len
            unpacker.feed(data)

            if received_len == orig_len:
                break

        # Get the data.
        for message in unpacker:
            sts = message.get("status", message.get(b"status"))
            if sts:
                if sts == "ok" or sts == b"ok":
                    return message
                else:
                    raise TensorForceError("RemoteEnvironment server error: {}".
                                           format(message.get("message", "not specified")))
            else:
                raise TensorForceError("Message without field 'status' received!")
        raise TensorForceError("No message encoded in data stream (data stream had len={})".
                               format(orig_len))
