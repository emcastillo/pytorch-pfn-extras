# Lazy Modules

Lazy modules are modules that 

Following modules are provided:

* pe.nn.LazyLinear
    * Module that behaves as `torch.nn.Linear` but `in_features` can be set to `None`.
* pe.nn.LazyConv1d, pte.nn.LazyConv2d, pte.nn.LazyConv3d
    * Module that behaves as `torch.nn.Conv[123]d` but `in_channels` can be set to `None`.

Credit: the original idea and implementation are brought by @nakago. Thank you!
