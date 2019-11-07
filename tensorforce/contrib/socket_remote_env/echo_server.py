import pickle


class EchoServer(object):
    '''Implement a simple echo server for sending data and instruction through a socket.
    '''

    def __init__(self, verbose=0):

        # the commands that are supported; should correspond to methods
        # in the RemoteEnvironmentServer class. Those commands can be
        # used by the RemoteEnvironmentClient.

        self.supported_requests = (
            # Put the simulation to the state from which learning begins.
            # If successfull RESET: 1RESET, fail empty
            'RESET',
            # Respond with the state of Simulation (some vector in state space)
            'STATE',
            # CONTROL: valuesCONTROL, success CONTROL: 1CONTROL, fail empty
            'CONTROL',
            # Evolve using the set control, success EVOLVE: 1EVOLVE, fail empty
            'EVOLVE',
            # Response to reward, sucess REWARD: valueREWARD, fail empty
            'REWARD',
            # Is the solver done? value 0 1, empty fail
            'TERMINAL',
            # time to close the socket
            'CLOSE',
            )

        self.verbose = verbose

    @staticmethod
    def decode_message(msg, verbose=1):
        msg = pickle.loads(msg)

        assert(isinstance(msg, (list,)))
        assert(len(msg) == 2)

        request = msg[0]
        data = msg[1]

        if verbose > 1:
            print("decode message --------------")
            print(" request:")
            print(request)
            print(" data:")
            print(data)
            print("-----------------------------")

        return request, data

    @staticmethod
    def encode_message(request, data, verbose=0):
        '''Encode data (a list) as message'''

        complete_message = [request, data]
        msg = pickle.dumps(complete_message)

        if verbose > 1:
            print("encode message --------------")
            print(" request:")
            print(request)
            print(" data:")
            print(data)

        return msg

    def handle_message(self, msg):
        '''Trigger action base on client message.'''

        request, data = EchoServer.decode_message(msg, verbose=self.verbose)

        if request not in self.supported_requests:
            RuntimeError("unknown request; no support for {}".format(request))
            return EchoServer.encode_message(request, [])

        # Dispatch to action which returns the data. This is what
        # children need to implement
        # so this calls the method of the EchoSolver (or the class derived from it)
        # such as RemoteEnvironmentServer
        result = getattr(self, request)(data)

        # Wrap
        return EchoServer.encode_message(request, result, verbose=self.verbose)
