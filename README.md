# pytorch-pfn-extras

Supplementary components to accelerate research and development in PyTorch.

## Installation

```sh
# Released version (hosted on PFN PyPI)
pip install pytorch-pfn-extras --extra-index-url https://pypi.pfn.io/simple

# Development version
pip install git+https://github.com/pfnet/pytorch-pfn-extras

### Optinal dependencies
# For PlotReport / VariableStatisticsPlot extensions
pip install matplotlib

# For IgniteExtensionsManager and example code
pip install pytorch-ignite torchvision
```

## Documentation

* [Extensions Manager](docs/extensions.md)
* [Reporter](docs/reporter.md)
* [Lazy Modules](docs/lazy.md)
* [Distributed Snapshot](docs/snapshot.md)

## Examples

* [Custom training loop](example/mnist.py)
* [Ignite integration](example/mnist.py)

## Contribution Guide

You can contribute to this project by sending a pull request.
After approval, the pull request will be merged by the reviewer.

Before making a contribution, please confirm that:

- Code quality stays consistent across the script, module or package.
- Code is covered by unit tests.
- API is maintainable.
