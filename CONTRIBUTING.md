# Contribution guide

Please always get in touch on [Gitter](https://gitter.im/tensorforce/community) before start working on a pull request, unless it is a smaller bug fix involving only a few lines of code.


### Code style

- [Google Python style guide](https://google.github.io/styleguide/pyguide.html)
- Maximum line length: 100 characters; tab size: 4 spaces
- There should be no PEP8 warnings (apart from E501 regarding line length)
- If arguments, when initializing objects / calling functions / specifying lists/dicts / etc, do not fit into the same line, should be in one (or multiple) separate tab-indented line(s), like this:

```python
super().__init__(
    states=states, actions=actions, l2_regularization=l2_regularization,
    parallel_interactions=parallel_interactions, config=config, saver=saver, summarizer=summarizer
)
```

- TensorFlow as well as Tensorforce-internal function calls should use named arguments wherever possible
- Binary operators should always be surrounded by a single space, so `z = x + y` instead of `z=x+y`
- Numbers should always be specified according to their intended type, so `1.0` as opposed to `1` in the case of floats, and vice versa for integers. For clarity, floats should furthermore add single leading/trailing zeros where necessary, so `1.0` instead of `1.` and `0.1` instead of `.1`.
- Line comments should generally be in a separate line preceding the line(s) they are commenting on, and not be added after the code as a suffix.
