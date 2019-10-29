import socket

# usually this will be in a simulation_baseline folder; then need to import the
# EchoServer

import sys
import os
cwd = os.getcwd()
sys.path.append(cwd + "/../")

from echo_server import EchoServer


class RemoteEnvironmentServer(EchoServer):

    def __init__(self,
                 tensorforce_environment,
                 host='localhost',
                 port=12230,
                 buffer_size=262144,
                 verbose=1):

        # tensorforce_environment should be a ready-to-use environment
        # host, port is where making available

        self.tensorforce_environment = tensorforce_environment
        self.state = None
        self.terminal = False
        self.reward = None
        self.nbr_reset = 0
        
        self.active_socket = False

        self.buffer_size = buffer_size

        EchoServer.__init__(self, verbose)

        # set up the socket
        socket_instance = socket.socket()

        if host is None:
            host = socket.gethostname()

        socket_instance.bind((host, port))

        socket_instance.listen(1)  # Buffer only one request
        self.active_socket = True

        connection = None

        while True:
            if connection is None:
                if verbose > 1:
                    print('[Waiting for connection...]')
                connection, address = socket_instance.accept()
                if verbose > 1:
                    print('Got connection from {}'.format(address))
            else:
                if verbose > 1:
                    print('[Waiting for request...]')
                message = connection.recv(self.buffer_size)

                response = self.handle_message(message)  # this is given by the EchoServer base class

                connection.send(response)

        if self.active_socket:
            socket_instance.close()
            self.active_socket = False

    def RESET(self, data):
        self.nbr_reset += 1
        self.state = self.tensorforce_environment.reset()
        return(1)

    def STATE(self, data):
        return(self.state)

    def TERMINAL(self, data):
        return(self.terminal)

    def REWARD(self, data):
        return(self.reward)

    def CONTROL(self, data):
        self.actions = data
        return(1)

    def EVOLVE(self, data):
        self.state, self.terminal, self.reward = self.tensorforce_environment.execute(self.actions)
        return(1)
        
    def CLOSE(self, data):
        if self.active_socket:
            socket_instance.close()
            self.active_socket = False
            
        return(1)
