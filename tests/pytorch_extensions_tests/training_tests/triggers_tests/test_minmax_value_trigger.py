import mock
import pytest

from pytorch_extensions.training import triggers
from pytorch_extensions import training


def _test_trigger(trigger, key, accuracies, expected, iter_per_epoch):
    optimizers = {'main': mock.MagicMock()}
    epochs = -(-len(expected) // iter_per_epoch)
    trainer = training.ExtensionsManager({}, optimizers, epochs, [])

    invoked_iterations = []

    for it, e in enumerate([0.] + accuracies):
        with trainer.run_iteration(iteration=it, epoch_size=iter_per_epoch):
            trainer.observation = {key: accuracies[it - 1]}
            if trigger(trainer):
                invoked_iterations.append(it)

    assert invoked_iterations == expected


@pytest.mark.parametrize(
    'trigger_type,iter_per_epoch,interval,accuracies,expected',
    [
        # interval = 1 iterations
        (triggers.MaxValueTrigger, 1, (1, 'iteration'),
         [0.5, 0.5, 0.4, 0.6], [1, 4]),
        (triggers.MinValueTrigger, 1, (1, 'iteration'),
         [0.5, 0.5, 0.4, 0.6], [1, 3]),
        # interval = 2 iterations
        (triggers.MaxValueTrigger, 1, (2, 'iteration'),
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 8]),
        (triggers.MinValueTrigger, 1, (2, 'iteration'),
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 6]),
        # interval = 2 iterations, unaligned resume
        (triggers.MaxValueTrigger, 1, (2, 'iteration'),
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 8]),
        (triggers.MinValueTrigger, 1, (2, 'iteration'),
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 6]),
        # interval = 1 epoch, 1 epoch = 2 iterations
        (triggers.MaxValueTrigger, 2, (1, 'epoch'),
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 8]),
        (triggers.MinValueTrigger, 2, (1, 'epoch'),
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 6]),
        # interval = 1 epoch, 1 epoch = 2 iterations, unaligned resume
        (triggers.MaxValueTrigger, 2, (1, 'epoch'),
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 8]),
        (triggers.MinValueTrigger, 2, (1, 'epoch'),
         [0.5, 0.5, 0.5, 0.5, 0.4, 0.4, 0.6, 0.6], [2, 6]),
    ]
)
def test_trigger(
        trigger_type, iter_per_epoch, interval, accuracies, expected):
    key = 'main/accuracy'
    trigger = trigger_type(key, trigger=interval)
    _test_trigger(trigger, key, accuracies, expected, iter_per_epoch)
