import torch

from pytorch_extensions.nn.modules.lazy import LazyInitializationMixin
from pytorch_extensions.nn.modules.lazy import UninitializedParameter


class _LazyConvNd(LazyInitializationMixin):

    _lazy_parameter_keys = ('weight',)

    def __init__(self, in_channels, *args, **kwargs):
        super(_LazyConvNd, self).__init__(
            0 if in_channels is None else in_channels, *args, **kwargs)
        if in_channels is None:
            self.in_channels = None
            self.weight = UninitializedParameter()

    def forward(self, input):
        if isinstance(self.weight, UninitializedParameter):
            self.in_channels = input.shape[1]
            if self.transposed:
                shape = (self.in_channels, self.out_channels // self.groups,
                         *self.kernel_size)
            else:
                shape = (self.out_channels, self.in_channels // self.groups,
                         *self.kernel_size)
            self.weight = torch.nn.Parameter(self.weight.new_empty(*shape))
            self.reset_parameters()
        return super(_LazyConvNd, self).forward(input)

    def reset_parameters(self):
        # Defer initialization of parameters until shape of all parameters
        # are ready.
        if self.lazy_parmeters_determined:
            super(_LazyConvNd, self).reset_parameters()


class LazyConv1d(_LazyConvNd, torch.nn.Conv1d):
    r"""Conv1d module with lazy weight initialization.

    When `in_channels` is `None`, it is determined at the first time of
    the forward step.
    """
    pass


class LazyConv2d(_LazyConvNd, torch.nn.Conv2d):
    r"""Conv2d module with lazy weight initialization.

    When `in_channels` is `None`, it is determined at the first time of
    the forward step.
    """
    pass


class LazyConv3d(_LazyConvNd, torch.nn.Conv3d):
    r"""Conv3d module with lazy weight initialization.

    When `in_channels` is `None`, it is determined at the first time of
    the forward step.
    """
    pass
