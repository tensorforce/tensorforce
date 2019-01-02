# Update notes

This file tracks all major updates and new features. As TensorForce is still in alpha, 
we are continuously implementing small updates and bug fixes, which will not
be tracked here in detail but through github issues.

04/11/18:
- States/actions of type `int` require a parameter `num_values` (instead of `num_actions`)
- `Embedding` layer argument `indices` renamed to `num_embeddings`
- `Flatten` layer replaced by `GlobalPooling` layer
- Baseline option `custom` renamed to `network`

- Environment states/actions???????
- New `Model` arguments: `l2_regularization`
- Layer argument `l1_regularization` removed
- `Input` layer renamed to `Retrieve` and argument `names` renamed to `tensors`
- `Output` layer renamed to `Register` and argument `name` renamed to `register`
- `Nonlinearity` layer argument `name` renamed to `activation`
- `Pool2d` layer argument `pooling_type` renamed to `pooling`
- `Dense` layer argument `skip` removed  ????
- `Linear` and `Dense` layer argument `size=None` not allowed  ????
- `Embedding` layer argument `num_embeddings` not required anymore
- `LayerBasedNetwork` renamed to `LayerbasedNetwork`
