run.py -- Runner
================


##### Required arguments

#1: **agent** (*string*) -- Agent (configuration JSON file, name, or library module)
<br>
#2: **environment** (*string*) -- Environment (name, configuration JSON file, or library module)


##### Optional arguments

###### Agent arguments

**-\-[n]etwork** (*string, default: not specified*) -- Network (configuration JSON file, name, or library module)

###### Environment arguments

**-\-[l]evel** (*string, default: not specified*) -- Level or game id, like `CartPole-v1`, if supported
<br>
**-\-[i]mport-modules** (*string, default: not specified*) -- Import comma-separated modules required for environment
<br>
**-\-visualize** (*bool, default: false*) -- Visualize agent--environment interaction, if supported

###### Runner arguments

**-\-[t]imesteps** (*int, default: not specified*) -- Number of timesteps
<br>
**-\-[e]pisodes** (*int, default: not specified*) -- Number of episodes
<br>
**-\-[m]ax-episode-timesteps** (*int, default: not specified*) -- Maximum number of timesteps per episode
<br>
**-\-mean-horizon** (*int, default: 10*) -- Number of timesteps/episodes for mean reward computation
<br>
**-\-e[v]aluation** (*bool, default: false*) -- Evaluation mode
<br>
**-\-[s]ave-best-agent** (*bool, default: false*) -- Save best-performing agent

###### Logging arguments

**-\-[r]epeat** (*int, default: 1*) -- Number of repetitions
<br>
**-\-[p]ath** (*string, default: not specified*) -- Logging path, directory plus filename without extension

**-\-seaborn** (*bool, default: false*) -- Use seaborn
