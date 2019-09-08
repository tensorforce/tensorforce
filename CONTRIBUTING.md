# Contribution guide

Please always get in touch on [Gitter](https://gitter.im/tensorforce/community) before start working on a pull request, unless it is a smaller bug fix or similar involving only a few lines of code.


### Code style

- [Google Python style guide](https://google.github.io/styleguide/pyguide.html)
- Maximum line length: 100 characters; tab size: 4 spaces
- There should be no PEP8 warnings (apart from E501 regarding line length)
- Arguments when initializing objects / calling functions / specifying lists/dicts / etc, if they do not fit into the same line, should be in one (or multiple) separate tab-indented line(s), like this:

```python
super().__init__(
    name=name, device=device, parallel_interactions=parallel_interactions,
    buffer_observe=buffer_observe, execution=execution, saver=saver, summarizer=summarizer,
    states=states, internals=internals, actions=actions, preprocessing=preprocessing,
    exploration=exploration, variable_noise=variable_noise, l2_regularization=l2_regularization
)
```

- TensorFlow as well as Tensorforce-internal function calls should use named arguments wherever possible
- Binary operators should always be surrounded by a single space, so `z = x + y` instead of `z=x+y`
- Numbers should always be specified according to their intended type, so `1.0` as opposed to `1` in the case of floats, and vice versa for integers. Floats should furthermore add single leading/trailing zeros where necessary, so `1.0` instead of `1.` and `0.1` instead of `.1`.
- Line comments should generally be in a separate line preceding the line(s) they are commenting on, and not be added after the code as a suffix.


### Naming conventions

- Tensor constants: `zero`, `zeros`, `one`, `ones`, `false`, `true`
- Tensor shape: `shape`; tensor rank ~ `len(shape)`: `rank`; tensor rank dimensions ~ `len(shape[n])`: `dims`
