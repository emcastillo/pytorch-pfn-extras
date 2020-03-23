import setuptools


setuptools.setup(
    name='pytorch-pfn-extras',
    description='Supplementary components to accelerate research and '
                'development in PyTorch.',
    version='0.1.2',
    install_requires=['numpy', 'torch'],
    extras_require={'test': ['pytest']},
    packages=[
        'pytorch_pfn_extras',
        'pytorch_pfn_extras.nn',
        'pytorch_pfn_extras.nn.modules',
        'pytorch_pfn_extras.training',
        'pytorch_pfn_extras.training.extensions',
        'pytorch_pfn_extras.training.triggers',
        'pytorch_pfn_extras.writing',
    ],
)
