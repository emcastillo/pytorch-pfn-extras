import mock
import pytest

from pytorch_extensions.training import triggers
from pytorch_extensions import training


def _test_trigger(trigger, key, accuracies, expected, iters_per_epoch):
    optimizers = {'main': mock.MagicMock()}
    max_epochs = -(-len(expected) // iters_per_epoch)
    trainer = training.ExtensionsManager(
        {}, optimizers, max_epochs, iters_per_epoch=iters_per_epoch)

    invoked_iterations = []

    for it, e in enumerate([0.] + accuracies):
        with trainer.run_iteration():
            trainer.observation = {key: accuracies[it - 1]}
            if trigger(trainer):
                invoked_iterations.append(it)

    assert invoked_iterations == expected


def _compare(best_value, new_value):
    return abs(new_value) < abs(best_value)


@pytest.mark.parametrize(
    'trigger_type,trigger_args,iters_per_epoch,accuracies,expected',
    [
        # interval = 1 iterations
        (triggers.MaxValueTrigger, ((1, 'iteration'),), 1,
         [0.5, 0.5, 0.4, 0.6], [1, 4]),
        (triggers.MinValueTrigger, ((1, 'iteration'),), 1,
         [0.5, 0.5, 0.4, 0.6], [1, 3]),
        # interval = 2 iterations
        (triggers.MaxValueTrigger, ((2, 'iteration'),), 1,
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 8]),
        (triggers.MinValueTrigger, ((2, 'iteration'),), 1,
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 6]),
        # interval = 2 iterations, unaligned resume
        (triggers.MaxValueTrigger, ((2, 'iteration'),), 1,
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 8]),
        (triggers.MinValueTrigger, ((2, 'iteration'),), 1,
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 6]),
        # interval = 1 epoch, 1 epoch = 2 iterations
        (triggers.MaxValueTrigger, ((1, 'epoch'),), 2,
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 8]),
        (triggers.MinValueTrigger, ((1, 'epoch'),), 2,
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 6]),
        # interval = 1 epoch, 1 epoch = 2 iterations, unaligned resume
        (triggers.MaxValueTrigger, ((1, 'epoch'),), 2,
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 8]),
        (triggers.MinValueTrigger, ((1, 'epoch'),), 2,
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 6]),

        # best_value trigger test
        (triggers.BestValueTrigger, (_compare, (1, 'iteration')), 2,
         [0.5, -0.5, -0.6, 0.6, 0.4, -0.4, -0.3, 0.3], [1, 5, 7]),
    ]
)
def test_minmax_trigger(
        trigger_type, trigger_args, iters_per_epoch, accuracies, expected):
    key = 'main/accuracy'
    trigger = trigger_type(key, *trigger_args)
    _test_trigger(trigger, key, accuracies, expected, iters_per_epoch)
