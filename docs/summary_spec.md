TensorForce: Details for "summary_spec" agent parameters
====================================================================

[![Docs](https://readthedocs.org/projects/tensorforce/badge)](http://tensorforce.readthedocs.io/en/latest/)
[![Gitter](https://badges.gitter.im/reinforceio/TensorForce.svg)](https://docs.google.com/forms/d/1_UD5Pb5LaPVUviD0pO0fFcEnx_vwenvuc00jmP2rRIc/)
[![Build Status](https://travis-ci.org/reinforceio/tensorforce.svg?branch=master)](https://travis-ci.org/reinforceio/tensorforce)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/reinforceio/tensorforce/blob/master/LICENSE)

summarizer
------------

TensorForce has the ability to record summary data for use with TensorBoard
as well STDIO and file export.  This is accomplished through dictionary
parameter called "summarizer" passed to the agent on initialization.

"summarizer" supports the following optional dictionary entries:

```eval_rst
+--------------+------------------------------------------------------------+
| Key          | Value                                                      |
+==============+============================================================+
| directory    | (str) Path to storage for TensorBoard summary data         |
+--------------+------------------------------------------------------------+
| steps        | (int) Frequency in steps between storage of summary data   |
+--------------+------------------------------------------------------------+
| seconds      | (int) Frequency in seconds to store summary data           |
+--------------+------------------------------------------------------------+
| labels       | (list) Requested Export, See "*LABELS*" section            |
+--------------+------------------------------------------------------------+
| meta\_dict   | (dict) For used with label "configuration"                 |
+--------------+------------------------------------------------------------+
```


LABELS
------
```eval_rst

+------------------------+---------------------------------------------------------+
| Entry                  | Data produced                                           |
+========================+=========================================================+
| losses                 | Training total-loss and "loss-without-regularization"   |
+------------------------+---------------------------------------------------------+
| total-loss             | Final calculated loss value                             |
+------------------------+---------------------------------------------------------+
| variables              | Network variables                                       |
+------------------------+---------------------------------------------------------+
| inputs                 | Equivalent to: ['states', 'actions', 'rewards']         |
+------------------------+---------------------------------------------------------+
| states                 | Histogram of input state space                          |
+------------------------+---------------------------------------------------------+
| actions                | Histogram of input action space                         |
+------------------------+---------------------------------------------------------+
| rewards                | Histogram of input reward space                         |
+------------------------+---------------------------------------------------------+
| gradients              | Histogram and scalar gradients                          |
+------------------------+---------------------------------------------------------+
| gradients\_histogram   | Variable gradients as histograms                        |
+------------------------+---------------------------------------------------------+
| gradients\_scalar      | Variable Mean/Variance of gradients as scalar           |
+------------------------+---------------------------------------------------------+
| regularization         | Regularization values                                   |
+------------------------+---------------------------------------------------------+
| **configuration**      | See *Configuration Export* for more detail              |
+------------------------+---------------------------------------------------------+
| configuration          | Export configuration to "TEXT" tab in TensorBoard       |
+------------------------+---------------------------------------------------------+
| print\_configuration   | Prints configuration to STDOUT                          |
+------------------------+---------------------------------------------------------+
```

```python
from tensorforce.agents import PPOAgent

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    states=...,
    actions=...,
    network=...,
    summarizer=dict(directory="./board/",
                        steps=50,
                        labels=['configuration',
                            'gradients_scalar',
                            'regularization',
                            'inputs',
                            'losses',
                            'variables']
                    ),      
    ...
)
```

Configuration Export
--------------------

Adding the "configuration" label will create a "TEXT" tab in TensorBoard
that contains all the parameters passed to the Agent.  By using the additional
"summarizer" dictionary key "meta_dict", custom keys and values can be added
to the data export.  The user may want to pass "Description", "Experiement #",
 "InputDataSet", etc.

If a key is already in use within TensorForce an error will be raised to
notify you to change the key value.  To use the custom feature, create a
dictionary with keys to export:
```python
from tensorforce.agents import PPOAgent

metaparams['MyDescription'] = "This experiment covers the first test ...."
metaparams['My2D'] = np.ones((9,9))   # 9x9 matrix of  1.0's
metaparams['My1D'] = np.ones((9))   # Column of 9   1.0's

# Create a Proximal Policy Optimization agent
agent = PPOAgent(
    states=...,
    actions=...,
    network=...,
    summarizer=dict(directory="./board/",
                        steps=50,
                        meta_dict=metaparams,  #Add custom keys to export
                        labels=['configuration',
                            'gradients_scalar',
                            'regularization',
                            'inputs',
                            'losses',
                            'variables']
                    ),      
    ...
)
```

Use the "print_configuration" label to export the configuration data to the
command line's STDOUT.
