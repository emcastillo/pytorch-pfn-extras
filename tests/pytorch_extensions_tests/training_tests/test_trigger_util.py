import mock
import pytest

from pytorch_extensions import training
from pytorch_extensions.training import trigger_util
from pytorch_extensions.training import triggers


@pytest.mark.parametrize(
    'iter_per_epoch,trigger_args,expected',
    [
        # Never fire trigger
        (2, None, [False, False, False, False, False, False, False]),

        # Interval trigger
        (2, (2, 'iteration'),
         [False, True, False, True, False, True, False]),
        (2, (2, 'epoch'),
         [False, False, False, True, False, False, False]),

        # Callable object
        (2, trigger_util.get_trigger(None),
         [False, False, False, False, False, False, False]),
        (2, triggers.IntervalTrigger(2, 'iteration'),
         [False, True, False, True, False, True, False]),
    ]
)
def test_get_trigger(iter_per_epoch, trigger_args, expected):
    optimizers = {'main': mock.MagicMock()}
    epochs = -(-len(expected) // iter_per_epoch)
    trainer = training.ExtensionsManager({}, optimizers, epochs, [])
    trigger = trigger_util.get_trigger(trigger_args)

    # before the first iteration, trigger should be False
    for it, e in enumerate([False] + expected):
        with trainer.run_iteration(iteration=it, epoch_size=iter_per_epoch):
            assert trigger(trainer) == e
