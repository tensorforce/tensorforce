import socket


def check_free_port(host, port, verbose=True):
    """Check if a given port is available."""
    sock = socket.socket()
    try:
        sock.bind((host, port))
        sock.close()
        if verbose:
            print("host {} on port {} is AVAIL".format(host, port))
        return(True)
    except:
        if verbose:
            print("host {} on port {} is BUSY".format(host, port))
        sock.close()
        return(False)


def check_ports_avail(host, list_ports, verbose=True):
    for crrt_port in list_ports:
        if not check_free_port(host, crrt_port, verbose=verbose):
            if verbose:
                print("Not all ports are available; quitting!")
            return(False)

        if verbose:
            print("all ports available")

        return(True)


def bash_check_avail(first_port, n_ports):
    host = socket.gethostname()
    list_ports = [ind_server + first_port for ind_server in range(n_ports)]

    if check_ports_avail(host, list_ports, verbose=False):
        print("T")
        return(True)

    else:
        print("F")
        return(False)
