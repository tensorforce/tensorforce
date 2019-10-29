#!/bin/bash

if [ $# -eq 0 ]; then
    echo "No arguments provided; -h for help"
    exit 1
fi

if [ "$1" == "-h" ]; then
    echo "A bash script to launch the parallel training automatically:"
    echo "- create a new tmux session"
    echo "- split it"
    echo "- launch the training in it"
    echo "Usage:"
    echo "bash script_launch_parallel.sh session_name first_port num_servers"
    exit 0

else
    if [ $# -eq 3 ]; then
        # all good
        :
    else
        echo "Wrong number of arguments, abort; see help with -h"
        exit 1
    fi
fi

# check that the tmux session name is free
found_session=$(tmux ls | grep $1 | wc -l)

if [ $found_session != 0 ]; then
    echo "Collision in session name!"
    echo "running:    tmux ls | grep $1"
    echo "returned    $(tmux ls | grep $1)"
    echo "choose a different session name!"
    exit 1
fi

# check that all ports are free
output=$(python3 -c "from utils import bash_check_avail; bash_check_avail($2, $3)")

if [ $output == "T" ]; then
    echo "Ports available, launch..."
else
    if [ $output == "F" ]; then
        echo "Abort; some ports are not avail"
        exit 0
    else
        echo "wrong output checking ports; abort"
        exit 1
    fi
fi

# if I went so far, all ports are free and the tmux is available: can launch!

# create our tmux
tmux new -s $1 -d

# split and split to get 4 windows
tmux split-window -h
tmux split-window -v
tmux split-window -t 0 -v

# launch everything
tmux send-keys -t 1 "htop" C-m

echo "Launching the servers. This takes a few seconds..."
tmux send-keys -t 3 "python3 launch_servers.py -p $2 -n $3"  C-m
let "n_sec_sleep = 10 * $3"
echo "Wait $n_sec_sleep secs for servers to start..."
sleep $n_sec_sleep

echo "Launched training!"
tmux send-keys -t 2 "python3 launch_parallel_training.py -p $2 -n $3"  C-m

# have a look at the training, from the still available first pane
tmux select-pane -t 0
tmux attach -t $1

exit 0
