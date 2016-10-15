class TensorForceException(Exception):
    pass


class TensorForceValueException(TensorForceException):
    pass


class ArgumentMustBePositiveException(TensorForceValueException):
    pass
