Update notes
------------

This file tracks all major updates and new features. As TensorForce is still in alpha, 
we are continuously implementing small updates and bug fixes, which will not
be tracked here in detail but through github issues.

04/11/18:
- States of type `int` require a parameter `num_states`
- `Embedding` layer argument `indices` renamed to `num_embeddings`
- `Flatten` layer replaced by `GlobalPooling` layer
- Baseline option `custom` renamed to `network`
