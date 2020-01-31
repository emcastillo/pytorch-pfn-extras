import inspect
import warnings

import torch


class LazyInitializationMixin(object):

    r"""A mixin for modules that lazily initialize buffers and parameters.

    Unlike regular modules, subclasses of this module can initialize
    buffers and parameters outside of the constructor (`__init__`).
    This allows you to, for example, initialize parameters in `forward`
    method to determine the shape of the weight based on the initial input.

    Be sure to run "dummy" forward once to initialize all parameters that
    should be trained, before passing `module.parameters()` to an optimizer;
    otherwise weights initialized after `module.parameters()` (e.g., in
    `forward` function) will never be trained.

    Note that lazy modules cannot validate if the shape is correct during
    deserialization.  Also note that the initial weights may become different
    from the original (non-lazy) module even if the random seed is manuall
    configured, as the order of initialization is different from the original
    one; especially, `module.cuda()` may cause the initialization to run on
    a GPU.

    The default value of lazy buffers and parameters are `torch.Tensor([])`
    and `UninitializedParameter()`, respectively.
    """

    # Subclasses must override these fields and list names of all buffers /
    # parameters that will be initialized lazily.
    _lazy_buffer_keys = []
    _lazy_parameter_keys = []

    def __init__(self, *args, **kwargs):
        self._lazy_ready = False

        super(LazyInitializationMixin, self).__init__(*args, **kwargs)

        for key in self._lazy_buffer_keys:
            self.register_buffer(key, torch.Tensor([]))
        for key in self._lazy_parameter_keys:
            self.register_parameter(key, UninitializedParameter())
        self._register_load_state_dict_pre_hook(self._lazy_load_hook)
        self._lazy_ready = True

    @property
    def lazy_parmeters_determined(self):
        r"""Returns if all lazy parameters are determined.

        Subclasses can perform parameters initialization after all lazy
        parameters are determined.  Note that this may be called during
        `__init__`.
        """
        return self._lazy_ready and all([
            not isinstance(getattr(self, x), UninitializedParameter)
            for x in self._lazy_parameter_keys])

    def state_dict(self, *args, **kwargs):
        # Exclude uninitialized parameter from serialization.
        destination = super(LazyInitializationMixin, self).state_dict(
            *args, **kwargs)
        for key in self._lazy_parameter_keys:
            if isinstance(getattr(self, key), UninitializedParameter):
                del destination[key]
        return destination

    def _lazy_load_hook(
            self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        for key in self._lazy_buffer_keys:
            self.register_buffer(key, state_dict[prefix + key])
        for key in self._lazy_parameter_keys:
            # The key may not exist in the loaded `state_dict` if the
            # original module was serialized before initializing lazy
            # parameters.
            prefix_key = prefix + key
            if prefix_key in state_dict:
                # The model was serialized after initialization.
                self.register_parameter(
                     key, torch.nn.Parameter(state_dict[prefix_key]))
            else:
                # The model was serialized before initialization.
                param = UninitializedParameter()
                self.register_parameter(key, param)
                state_dict[prefix_key] = param


class UninitializedParameter(torch.nn.Parameter):

    def __repr__(self):
        return 'Uninitialized lazy parameter'

    @property
    def is_leaf(self):
        # Hacky workaround to detect use of uninitialized lazy parameters.
        # This overrides `is_leaf` attribute which should always be `True`
        # for parameters; optimizers check for this attribute and raise an
        # error if non-leaf tensors are detected.
        frame = inspect.currentframe()
        if frame.f_back.f_globals['__package__'].startswith('torch.optim'):
            warnings.warn('''
    Use of uninitialized lazy parameter in Optimizer has been detected.
    Maybe you forgot to run forward before passing `module.parameters()` to the optimizer?''')  # NOQA
        return True
