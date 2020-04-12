# pytorch-pfn-extras

Supplementary components to accelerate research and development in PyTorch.

## Installation

```sh
# Released version (hosted on PFN PyPI)
pip install pytorch-pfn-extras --extra-index-url https://pypi.pfn.io/simple

# Development version
pip install git+ssh://git@github.pfidev.jp/DLFW/pytorch-pfn-extras

### Optinal dependencies
# For PlotReport / VariableStatisticsPlot extensions
pip install matplotlib

# For IgniteExtensionsManager and example code
pip install pytorch-ignite torchvision
```

### Tips for FlexCI

To use `pytorch-pfn-extras` (PPE) in FlexCI projects, you need to install with `pip install --proxy=http://corp-proxy:3128`.
See [FlexCI Tips](https://docs.google.com/document/d/1sXCtbNxhcs91rIo5mBimVLX3Osoq98FQbRjzkzxb2EI/edit#heading=h.fw24rofb0pcr) for details.

This issue will be resolved once the library is released as an OSS.

## Documentation

* [Extensions Manager](docs/extensions.md)
* [Reporting](docs/reporting.md)
* [Lazy Modules](docs/lazy.md)
* [Distributed Snapshot](docs/snapshot.md)
* [Config System](docs/config.md)

## Examples

* [Custom training loop](example/mnist.py)
* [Ignite integration](example/ignite-mnist.py)

## Contribution Guide

You can contribute to this project by sending a pull request.
After approval, the pull request will be merged by the reviewer.

Before making a contribution, please confirm that:

- Code quality stays consistent across the script, module or package.
- Code is covered by unit tests.
- API is maintainable.
