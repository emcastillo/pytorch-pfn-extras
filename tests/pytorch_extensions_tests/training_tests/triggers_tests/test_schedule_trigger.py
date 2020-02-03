import mock
import pytest

from pytorch_extensions import training
from pytorch_extensions.training import triggers


def expected_finished(pos, num):
    return [i >= pos for i in range(num)]


@pytest.mark.parametrize(
    'iters_per_epoch,schedule,expected,finished',
    [
        # single iteration
        (2, (2, 'iteration'),
         [False, True, False, False, False, False, False],
         expected_finished(1, 7)),
        # multiple iteration
        (2, ([2, 4], 'iteration'),
         [False, True, False, True, False, False, False],
         expected_finished(3, 7)),
        # single epoch
        (3, (1, 'epoch'), [False, False, True, False, False, False, False],
         expected_finished(2, 7)),
        # multiple epoch
        (3, ([1, 2], 'epoch'), [False, False, True, False, False, True, False],
         expected_finished(5, 7)),
        # single fractional epoch
        (2, (1.5, 'epoch'), [False, False, True, False, False, False, False],
         expected_finished(2, 7)),
        # multiple fractional epoch
        (2, ([1.5, 2.5], 'epoch'),
         [False, False, True, False, True, False, False],
         expected_finished(4, 7)),
        # TODO(imanishi): Restore these tests after supported.
        # # single unaligned epoch
        # (2.5, (1, 'epoch'), [False, False, True, False, False, False, False],
        #  expected_finished(2, 7)),
        # # multiple unaligned epoch
        # (2.5, ([1, 2], 'epoch'),
        #  [False, False, True, False, True, False, False],
        #  expected_finished(4, 7)),
        # # single tiny epoch
        # (0.5, (1, 'epoch'), [True, False, False, False, False, False, False],
        #  expected_finished(0, 7)),
        # # multiple tiny epoch
        # (0.5, ([1, 2], 'epoch'),
        #  [True, False, False, False, False, False, False],
        #  expected_finished(0, 7)),
    ]
)
def test_trigger(iters_per_epoch, schedule, expected, finished):
    optimizers = {'main': mock.MagicMock()}
    max_epochs = -(-len(expected) // iters_per_epoch)
    trainer = training.ExtensionsManager(
        {}, optimizers, max_epochs, iters_per_epoch=iters_per_epoch)
    trigger = triggers.ManualScheduleTrigger(*schedule)

    for (e, f) in zip([False] + expected, [False] + finished):
        with trainer.run_iteration():
            assert trigger(trainer) == e
            assert trigger.finished == f


def test_invalid_unit():
    with pytest.raises(ValueError):
        triggers.ManualScheduleTrigger(1, 'day')
