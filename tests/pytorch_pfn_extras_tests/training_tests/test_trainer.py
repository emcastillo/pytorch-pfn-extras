import pytest

import torch
from torch import nn

import pytorch_pfn_extras as ppe
from pytorch_pfn_extras import training
from pytorch_pfn_extras.training import extensions


class MyModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1 = torch.nn.Linear(20, 15).to(device)
        self.l2 = torch.nn.Linear(15, 10)

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y.to('cpu'))
        return y


class MyModelWithLossFn(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.L1Loss()

    def forward(self, x, t):
        y = self.model(x).to(x.device)
        prefix = 'train' if self.training else 'val'
        loss = self.loss_fn(y, t)
        ppe.reporting.report({f'{prefix}/loss': loss})
        return loss


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_extensions_manager_extensions(device):
    model = MyModel(device)
    model_with_loss_fn = MyModelWithLossFn(model)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    data = [(torch.rand(20,), torch.rand(10,)) for i in range(10)]
    evaluator = training.create_evaluator(device, model_with_loss_fn)

    trainer_extensions = [
        extensions.LogReport(trigger=(10, 'iteration')),
        extensions.ProgressBar(update_interval=2),
        extensions.PrintReport(
            [
                'epoch',
                'iteration',
                'train/loss',
                'val/loss',
                'val/accuracy',
                'elapsed_time',
                'time',
            ]
        ),
    ]

    trainer = training.create_trainer(
        device,
        model_with_loss_fn,
        optim,
        20,
        iters_per_epoch=10,
        evaluator=evaluator,
        extensions=trainer_extensions,
    )
    trainer.run(data, data)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
