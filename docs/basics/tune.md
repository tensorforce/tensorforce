tune.py -- Hyperparameter tuner
===============================


##### Required arguments

#1: **environment** (*string*) -- Environment (name, configuration JSON file, or library module)


##### Optional arguments

**-\-[l]evel** (*string, default: not specified*) -- Level or game id, like `CartPole-v1`, if supported
<br>
**-\-[m]ax-repeats** (*int, default: 1*) -- Maximum number of repetitions
<br>
**-\-[n]um-iterations** (*int, default: 1*) -- Number of BOHB iterations
<br>
**-\-[d]irectory** (*string, default: "tuner"*) -- Output directory
<br>
**-\-[r]estore** (*string, default: not specified*) -- Restore from given directory
<br>
**-\-id** (*string, default: "worker"*) -- Unique worker id
