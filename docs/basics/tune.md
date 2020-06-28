tune.py -- Hyperparameter tuner
===============================


###### Environment arguments

**-\-[e]nvironment** (*string*, **required**) -- Environment (name, configuration JSON file, or library module)
<br>
**-\-[l]evel** (*string, default: not specified*) -- Level or game id, like `CartPole-v1`, if supported
<br>
**-\-[m]ax-episode-timesteps** (*int, default: not specified*) -- Maximum number of timesteps per episode
<br>
**-\-import-modules** (*string, default: not specified*) -- Import comma-separated modules required for environment


###### Runner arguments

**-\-episodes [n]** (*int*, **required**) -- Number of episodes
<br>
**-\-num-[p]arallel** (*int, default: no parallel execution*) -- Number of environment instances to execute in parallel


##### Tuner arguments

**-\-max-[r]epeats** (*int, default: 10*) -- Maximum number of repetitions
<br>
**-\-num-[i]terations** (*int, default: 1*) -- Number of BOHB iterations
<br>
**-\-[d]irectory** (*string, default: "tuner"*) -- Output directory
<br>
**-\-restore** (*string, default: not specified*) -- Restore from given directory
<br>
**-\-id** (*string, default: "worker"*) -- Unique worker id
