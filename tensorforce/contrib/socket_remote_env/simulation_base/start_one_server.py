"""
Start the current environment (i.e. the simulation described by the code contained
in this folder) and make it available through a server.
"""

import argparse
import socket

import sys
import os

from RemoteEnvironmentServer import RemoteEnvironmentServer
from env import resume_env

ap = argparse.ArgumentParser()

ap.add_argument("-p", "--port", required=True, help="the port to use", type=int)
ap.add_argument("-t", "--host", default="None", help="the host; default is local host; string either internet domain or IPv4", type=str)

args = vars(ap.parse_args())

port = args["port"]
host = args["host"]

with open('rank', 'r') as fh:
    rank_line = fh.readline()
    rank = int(rank_line)
    print("This is the simulation of rank {}".format(rank))

if host == 'None':
    host = 'localhost'


def check_free_port(host, port, verbose=True):
    """Check if a given port is available."""
    sock = socket.socket()
    try:
        sock.bind((host, port))
        sock.close()
        print("host {} on port {} is AVAIL".format(host, port))
        return(True)
    except:
        print("host {} on port {} is BUSY".format(host, port))
        sock.close()
        return(False)


if not check_free_port(host, port):
    print("the port is not available; quitting!")
    quit()


def launch_server(host, port, verbose=0):
    tensorforce_environment = resume_env(rank)    
        
    RemoteEnvironmentServer(tensorforce_environment, host=host, port=port,
            verbose=verbose)


launch_server(host, port)
