tune.py -- Hyperparameter tuner
===============================

Uses the [BOHB optimizer (Bayesian Optimization and Hyperband)](https://github.com/automl/HpBandSter) internally.


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

**-\-[r]uns-per-round** (*string, default: 1,2,5,10*) -- Comma-separated number of runs per optimization round, each with a successively smaller number of candidates
<br>
**-\-[s]election-factor** (*int, default: 3*) -- Selection factor n, meaning that one out of n candidates in each round advances to the next optimization round
<br>
**-\-num-[i]terations** (*int, default: 1*) -- Number of optimization iterations, each consisting of a series of optimization rounds with an increasingly reduced candidate pool
<br>
**-\-[d]irectory** (*string, default: "tuner"*) -- Output directory
<br>
**-\-restore** (*string, default: not specified*) -- Restore from given directory
<br>
**-\-id** (*string, default: "worker"*) -- Unique worker id
