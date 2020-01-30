import torch

import pytorch_extensions as pe


def test_observe_value():
    lr = 0.1
    manager = pe.training.ExtensionsManager(
        {}, [], 1, [],
        iters_per_epoch=1)
    extension = pe.training.extensions.observe_value('lr', lambda x: lr)
    manager.extend(extension)
    with manager.run_iteration():
        pass

    assert manager.observation['lr'] == lr


def test_observe_lr():
    lr = 0.01
    manager = pe.training.ExtensionsManager(
        {}, [], 1, [],
        iters_per_epoch=1)
    optimizer = torch.optim.Adam({torch.nn.Parameter()}, lr=lr)
    extension = pe.training.extensions.observe_lr(optimizer)
    manager.extend(extension)
    with manager.run_iteration():
        pass

    assert manager.observation['lr'] == lr
