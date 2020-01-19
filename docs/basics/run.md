run.py -- Runner
================


###### Agent arguments

**-\-[a]gent** (*string*, **required** *unless "socket-server" remote mode*) -- Agent (name, configuration JSON file, or library module)
<br>
**-\-[n]etwork** (*string, default: not specified*) -- Network (name, configuration JSON file, or library module)


###### Environment arguments

**-\-[e]nvironment** (*string*, **required** *unless "socket-client" remote mode*) -- Environment (name, configuration JSON file, or library module)
<br>
**-\-[l]evel** (*string, default: not specified*) -- Level or game id, like `CartPole-v1`, if supported
<br>
**-\-[m]ax-episode-timesteps** (*int, default: not specified*) -- Maximum number of timesteps per episode
<br>
**-\-visualize** (*bool, default: false*) -- Visualize agent--environment interaction, if supported
<br>
**-\-visualize-directory** (*bool, default: not specified*) -- Directory to store videos of agent--environment interaction, if supported
<br>
**-\-import-modules** (*string, default: not specified*) -- Import comma-separated modules required for environment


###### Parallel execution arguments

**-\-num-parallel** (*int, default: no parallel execution*) -- Number of environment instances to execute in parallel
<br>
**-\-batch-agent-calls** (*bool, default: false*) -- Batch agent calls for parallel environment execution
<br>
**-\-sync-timesteps** (*bool, default: false*) -- Synchronize parallel environment execution on timestep-level
<br>
**-\-sync-episodes** (*bool, default: false*) -- Synchronize parallel environment execution on episode-level
<br>
**-\-remote** (*str, default: local execution*) -- Communication mode for remote environment execution of parallelized environment execution: *"multiprocessing"* | *"socket-client"* | *"socket-server"*. In case of *"socket-server"*, runs environment in server communication loop until closed.
<br>
**-\-blocking** (*bool, default: false*) -- Remote environments should be blocking
<br>
**-\-host** (*str, only for "socket-client" remote mode*) -- Socket server hostname(s) or IP address(es), single value or comma-separated list
<br>
**-\-port** (*str, only for "socket-client/server" remote mode*) -- Socket server port(s), single value or comma-separated list, increasing sequence if single host and port given


###### Runner arguments

**-\-e[v]aluation** (*bool, default: false*) -- Run environment (last if multiple) in evaluation mode
<br>
**-\-e[p]isodes** (*int, default: not specified*) -- Number of episodes
<br>
**-\-[t]imesteps** (*int, default: not specified*) -- Number of timesteps
<br>
**-\-[u]pdates** (*int, default: not specified*) -- Number of agent updates
<br>
**-\-mean-horizon** (*int, default: 1*) -- Number of episodes progress bar values and evaluation score are averaged over
<br>
**-\-save-best-agent** (*bool, default: false*) -- Directory to save the best version of the agent according to the evaluation score

###### Logging arguments

**-\-[r]epeat** (*int, default: 1*) -- Number of repetitions
<br>
**-\-path** (*string, default: not specified*) -- Logging path, directory plus filename without extension

**-\-seaborn** (*bool, default: false*) -- Use seaborn
