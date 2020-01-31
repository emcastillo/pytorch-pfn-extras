import torch

from pytorch_extensions.nn.modules.lazy import UninitializedParameter
from pytorch_extensions.nn.modules.lazy import LazyInitializationMixin


class LazyLinear(LazyInitializationMixin, torch.nn.Linear):
    """Linear module with lazy weight initialization.

    When `in_features` is `None`, it is determined at the first time of
    the forward step.
    """

    _lazy_parameter_keys = ('weight',)

    def __init__(self, in_features, *args, **kwargs):
        super(LazyLinear, self).__init__(
            0 if in_features is None else in_features, *args, **kwargs)
        if in_features is None:
            self.in_features = None
            self.weight = UninitializedParameter()

    def forward(self, input):
        if isinstance(self.weight, UninitializedParameter):
            self.in_features = input.shape[-1]
            self.weight = torch.nn.Parameter(self.weight.new_empty(
                self.out_features, self.in_features))
            self.reset_parameters()
        return super(LazyLinear, self).forward(input)

    def reset_parameters(self):
        # Defer initialization of parameters until shape of the parameter
        # is determiend.
        if self.lazy_parmeters_determined:
            super(LazyLinear, self).reset_parameters()
