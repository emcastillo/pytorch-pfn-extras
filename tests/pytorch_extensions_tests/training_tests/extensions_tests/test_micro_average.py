import numpy

import pytorch_extensions as pe


def test_run():
    epoch_size = 5
    trigger_iters = 3
    data_shape = (4, trigger_iters)
    data_total = numpy.random.randint(7, 32, size=data_shape)
    data_correct = numpy.random.randint(data_total + 1)
    manager = pe.training.ExtensionsManager(
        {}, [], 100, [])

    extension = pe.training.extensions.MicroAverage(
        'main/correct', 'main/total', 'main/accuracy',
        (trigger_iters, 'iteration'))
    manager.extend(extension, (1, 'iteration'))

    for i, js in enumerate(numpy.ndindex(data_shape)):
        with manager.run_iteration(
                iteration=i, epoch_size=epoch_size):
            pe.reporter.report({
                'main/correct': data_correct[js],
                'main/total': data_total[js],
            })
        assert (
            # average is computed every trigger_iters
            ('main/accuracy' in manager.observation)
            == (js[1] == trigger_iters - 1))