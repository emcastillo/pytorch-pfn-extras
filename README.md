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

FlexCI (`ci.pfn.io`) cannot access PFN PyPI (`pypi.pfn.io`) nor GHE.
To use pytorch-pfn-extras (PPE) in FlexCI projects, download it using [gsutil](https://cloud.google.com/storage/docs/gsutil_install).

```
# `gsutil` command is pre-installed in the FlexCI environment.
# This creates `pytorch-pfn-extras` directory which works like a local PyPI repository.
gsutil -m cp -r gs://chainer-pfn-private-ci/pytorch-pfn-extras .

# Then install PPE using the local `pytorch-pfn-extras` directory.
pip install --find-links=pytorch-pfn-extras/index.html pytorch-pfn-extras
```

This issue will be resolved once the library is released as an OSS.

## Documentation

* [Extensions Manager](docs/extensions.md)
* [Reporting](docs/reporting.md)
* [Lazy Modules](docs/lazy.md)
* [Distributed Snapshot](docs/snapshot.md)

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
