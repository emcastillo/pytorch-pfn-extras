import torch


class Converter:

    """Base class of converters.

    Converters receive batched data retrieved from iterators and perform
    arbitrary transforms as well as device transfer.

    Implementation should override the ``__call__`` method.

    .. seealso::
        :meth:`pytorch_extensions.convert.converter` --- a decorator
        to turn a converter function into a ``Converter`` instance.

    """

    def __call__(self, batch, device):
        """Performs conversion.

        Args:
            batch:
                A batch. The type and value are arbitrary, depending on usage.
            device(~torch.device):
                Device to which the converter is expected to send the batch.

        Returns: A converted batch.
        """
        raise NotImplementedError(
            'Concrete class must implement __call__.')


class _ArbitraryCallableConverter(Converter):
    """Converter to wrap a callable with arbitrary arguments.

    This class accepts arbitrary arguments and pass-through to the underlying
    callable, with device argument replaced.
    """

    def __init__(self, base_callable):
        if not callable(base_callable):
            raise TypeError(
                'Can only wrap a callable. Actual: {}'.format(
                    type(base_callable)))
        self.base_callable = base_callable

    def __call__(self, *args, **kwargs):
        base_callable = self.base_callable

        # Normalize the 'device' argument
        if len(args) >= 2:
            # specified as a positional argument
            args = list(args)
            args[1] = _get_device(args[1])
        elif 'device' in kwargs:
            kwargs['device'] = _get_device(kwargs['device'])
        return base_callable(*args, **kwargs)


def converter():
    """Decorator to make a converter.

    This decorator turns a converter function into a
    :class:`pytorch_extensions.convert.Converter` class instance,
    which also is a callable.
    This is required to use the converter function from an old module that
    does not support :class:`torch.device` instances
    (See the **Device argument conversion** section below).

    .. rubric:: Requirements of the target function

    The target converter function must accept two positional arguments:
    a batch and a device, and return a converted batch.

    The type of the device argument is :class:`torch.device`.

    The types and values of the batches (the first argument and the return
    value) are not specified: they depend on how the converter is used (e.g.
    by updaters).

    .. admonition:: Example

        >>> @pytorch_extensions.training.convert.converter()
        ... def custom_converter(batch, device):
        ...     assert isinstance(device, torch.device)
        ...     # do something with batch...
        ...     return batch.to(device)

    .. rubric:: Device argument conversion

    For backward compatibility, the decorator wraps
    the function so that if the converter is called with the device argument
    with ``int`` type, it is converted to a :class:`torch.device`
    instance before calling the original function. The ``int`` value indicates
    the CUDA device of the cupy backend.

    Without the decorator, the converter cannot support ChainerX devices.
    If the batch were requested to be converted to ChainerX with such
    converters, :class:`RuntimeError` will be raised.

    """

    def wrap(func):
        return _ArbitraryCallableConverter(func)

    return wrap


def _call_converter(converter, batch, device):
    # Calls the converter.
    # Converter can be either new-style (accepts torch.device) or
    # old-style (accepts int as device).
    assert device is None or isinstance(device, torch.device)

    if isinstance(converter, Converter):
        # New-style converter
        return converter(batch, device)
    return converter(batch, device)


def to_device(device, x):
    """Send an array to a given device.

    This method sends a given array to a given device.
    You can also use this method in a custom converter method used in
    :class:`~pytorch_extensions.training.Extension`
    such as :class:`~pytorch_extensions.training.extensions.Evaluator`.

    Args:
        device (None or str or :class:`torch.device`): A device to
            which an array is sent. If it is``None``, an array
            is left in the original device.
        x (:ref:`ndarray`): An array to send.

    Returns:
        Converted array.

    """
    device = _get_device(device)

    if device is None:
        return x
    return x.to(device)


def _get_device(device_spec):
    # Converts device specificer to a torch.device instance.
    # this function supports None
    if device_spec is None:
        return None
    return torch.device(device_spec)


@converter()
def transfer_data(batch, device):
    return tuple([elem.to(device) for elem in batch])
