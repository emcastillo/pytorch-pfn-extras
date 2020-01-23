import torch


class LazyInitializationMixin(object):

    r"""Module that lazily initialize buffers and parameters.

    Unlike regular modules, subclasses of this module can initialize
    buffers and parameters outside of the constructor (`__init__`).
    This allows you to, for example, initialize parameters in `forward`
    method to determine the shape of the weight based on the initial input.

    Note that lazy modules cannot validate if the shape is correct during
    deserialization.  Also note that the initial weights may become different
    from the original (non-lazy) module even if the random seed is manuall
    configured, as the order of initialization is different from the original
    one; especially, `module.cuda()` may cause the initialization to run on
    a GPU.

    The default value of lazy buffers and parameters are `torch.Tensor([])`
    and `None`, respectively.
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
            self.register_parameter(key, None)
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
            getattr(self, x) is not None for x in self._lazy_parameter_keys])

    def _lazy_load_hook(self, state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs):
        for key in self._lazy_buffer_keys:
            self.register_buffer(key, state_dict[prefix + key])
        for key in self._lazy_parameter_keys:
            # As `state_dict()` does not serialize buffers and parameters
            # whose value is `None`, the key may not exist in the loaded
            # `state_dict` if the original module was serialized before
            # initializing lazy parameters.
            prefix_key = prefix + key
            if prefix_key in state_dict:
                # Regardless of the current state, initialize the parameter
                # with the loaded one.
                self.register_parameter(
                     key, torch.nn.Parameter(state_dict[prefix_key]))
            elif getattr(self, key) is not None:
                # The parameter is already initialized; revert to the
                # uninitialized state.
                self.register_parameter(key, None)
