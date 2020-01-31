import mock
import pytest

from pytorch_extensions import training
from pytorch_extensions.training import triggers


_argnames = ','.join(['iters_per_epoch', 'interval', 'expected'])


_argvalues = [
    # iteration
    (5, (2, 'iteration'), [False, True, False, True, False, True, False]),
    # basic epoch
    (1, (3, 'epoch'), [False, False, True, False, False, True, False]),
    # fractional epoch
    (2, (1.5, 'epoch'), [False, False, True, False, False, True, False]),
]


@pytest.mark.parametrize(_argnames, _argvalues)
def test_trigger(iters_per_epoch, interval, expected):
    optimizers = {'main': mock.MagicMock()}
    epochs = -(-len(expected) // iters_per_epoch)
    trainer = training.ExtensionsManager(
        {}, optimizers, epochs,
        iters_per_epoch=iters_per_epoch)
    trigger = triggers.IntervalTrigger(*interval)

    # before the first iteration, trigger should be False
    for iteration, e in enumerate([False] + expected):
        with trainer.run_iteration():
            assert trigger(trainer) == e


@pytest.mark.parametrize(_argnames, _argvalues)
def test_str(iters_per_epoch, interval, expected):
    trigger = triggers.IntervalTrigger(*interval)

    expected = 'IntervalTrigger({}, \'{}\')'.format(*interval)
    actual = str(trigger)

    assert expected == actual, 'Expected "{}" == "{}"'.format(expected, actual)


def test_invalid_unit():
    with pytest.raises(ValueError):
        triggers.IntervalTrigger(1, 'day')
