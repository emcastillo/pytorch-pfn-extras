import setuptools


setuptools.setup(
    name='pytorch_extensions',
    description='Torch port of several chainer extensions',
    version='0.0.0',
    install_requires=['numpy', 'torch'],
    extras_require={'test': ['pytest']},
    packages=[
        'pytorch_extensions',
        'pytorch_extensions.nn',
        'pytorch_extensions.nn.modules',
        'pytorch_extensions.training',
        'pytorch_extensions.training.extensions',
        'pytorch_extensions.training.triggers',
    ],
)
