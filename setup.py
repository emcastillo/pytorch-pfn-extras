import setuptools


setuptools.setup(
    name='pytorch_pfn_extras',
    description='Torch port of several chainer extensions',
    version='0.0.0',
    install_requires=['numpy', 'torch'],
    extras_require={'test': ['pytest']},
    packages=[
        'pytorch_pfn_extras',
        'pytorch_pfn_extras.nn',
        'pytorch_pfn_extras.nn.modules',
        'pytorch_pfn_extras.training',
        'pytorch_pfn_extras.training.extensions',
        'pytorch_pfn_extras.training.triggers',
    ],
)
