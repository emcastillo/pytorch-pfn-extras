import pytest

from pytorch_extensions import training
from pytorch_extensions.training import triggers


_argvalues = [
    # iteration
    (5, (2, 'iteration'), [False, True, False, True, False, True, False], 4),
    # basic epoch
    (1, (3, 'epoch'), [False, False, True, False, False, True, False], 4),
    # fractional epoch
    (2, (1.5, 'epoch'), [False, False, True, False, False, True, False], 4),
]


@pytest.mark.parametrize(
    'iters_per_epoch,interval,expected,resume', _argvalues)
def test_trigger(iters_per_epoch, interval, expected, resume):
    trainer = training.ExtensionsManager(
        {}, [], 100, iters_per_epoch=iters_per_epoch)
    trigger = triggers.IntervalTrigger(*interval)

    # before the first iteration, trigger should be False
    for e in [False] + expected:
        with trainer.run_iteration():
            assert trigger(trainer) == e


@pytest.mark.parametrize(
    'iters_per_epoch,interval,expected,resume', _argvalues)
def test_resumed_trigger(iters_per_epoch, interval, expected, resume):
    trainer = training.ExtensionsManager(
        {}, [], 100, iters_per_epoch=iters_per_epoch)
    trigger = triggers.IntervalTrigger(*interval)

    # before the first iteration, trigger should be False
    for e in [False] + expected[:resume]:
        with trainer.run_iteration():
            assert trigger(trainer) == e

    state = trigger.state_dict()
    new_trigger = triggers.IntervalTrigger(*interval)
    new_trigger.load_state_dict(state)

    for e in expected[resume:]:
        with trainer.run_iteration():
            assert new_trigger(trainer) == e


@pytest.mark.parametrize(
    'iters_per_epoch,interval,expected,resume', _argvalues)
def test_str(iters_per_epoch, interval, expected, resume):
    trigger = triggers.IntervalTrigger(*interval)

    expected = 'IntervalTrigger({}, \'{}\')'.format(*interval)
    actual = str(trigger)

    assert expected == actual, 'Expected "{}" == "{}"'.format(expected, actual)


def test_invalid_unit():
    with pytest.raises(ValueError):
        triggers.IntervalTrigger(1, 'day')
