import mock
import pytest

from pytorch_extensions import training
from pytorch_extensions.training import triggers


_argnames = ','.join(['iter_per_epoch', 'interval', 'expected'])


_argvalues = [
    # iteration
    (5, (2, 'iteration'), [False, True, False, True, False, True, False]),
    # basic epoch
    (1, (3, 'epoch'), [False, False, True, False, False, True, False]),
    # fractional epoch
    (2, (1.5, 'epoch'), [False, False, True, False, False, True, False]),
    # unaligned epoch
    (2.5, (1, 'epoch'), [False, False, True, False, True, False, False]),
    # tiny epoch
    (0.5, (1, 'epoch'), [True, True, True, True, True, True, True]),
]


@pytest.mark.parametrize(_argnames, _argvalues)
def test_trigger(iter_per_epoch, interval, expected):
    optimizers = {'main': mock.MagicMock()}
    epochs = -(-len(expected) // iter_per_epoch)
    trainer = training.ExtensionsManager({}, optimizers, epochs, [])
    trigger = triggers.IntervalTrigger(*interval)

    # before the first iteration, trigger should be False
    for iteration, e in enumerate([False] + expected):
        epoch = iteration // iter_per_epoch
        with trainer.run_iteration(
                epoch=epoch, iteration=iteration, epoch_size=iter_per_epoch):
            assert trigger(trainer) == e


@pytest.mark.parametrize(_argnames, _argvalues)
def test_str(iter_per_epoch, interval, expected):
    trigger = triggers.IntervalTrigger(*interval)

    expected = 'IntervalTrigger({}, \'{}\')'.format(*interval)
    actual = str(trigger)

    assert expected == actual, 'Expected "{}" == "{}"'.format(expected, actual)


def test_invalid_unit():
    with pytest.raises(ValueError):
        triggers.IntervalTrigger(1, 'day')
