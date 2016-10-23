class TensorForceError(Exception):
    pass


class TensorForceValueError(TensorForceError):
    pass


class ArgumentMustBePositiveError(TensorForceValueError):
    pass


class ConfigError(TensorForceValueError):
    pass
