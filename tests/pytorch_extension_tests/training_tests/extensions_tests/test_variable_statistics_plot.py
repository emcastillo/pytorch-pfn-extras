import os

import numpy
import pytest
import torch
import six

import pytorch_extensions as pe


@pytest.fixture(scope="module")
def matplotlib():
    try:
        import matplotlib
        matplotlib.use('Agg')
        return matplotlib
    except ImportError:
        pytest.skip('matplotlib is not installed')


def test_run_and_save_plot(matplotlib):
    filename = 'variable_statistics_plot_test.png'
    iterations = 2
    extension_trigger = (1, 'iteration')
    manager = pe.training.ExtensionsManager({}, [], 2, [])

    x = torch.rand(1, 2, 3)
    extension = pe.training.extensions.VariableStatisticsPlot(
        x, trigger=extension_trigger, filename=filename)
    manager.extend(extension, extension_trigger)

    # In the following we explicitly use plot_report._available instead of
    # PlotReport.available() because in some cases `test_available()` fails
    # because it sometimes does not raise UserWarning despite
    # matplotlib is not installed (this is due to the difference between
    # the behavior of unittest in python2 and that in python3).
    try:
        for i in range(iterations):
            cur_it = 1+i
            with manager.run_iteration(iteration=cur_it, epoch_size=2):
                pass
    finally:
        os.remove(os.path.join(manager.out, filename))


def test_reservoir_size():
    shape = (2, 7, 3)
    n = 5
    reservoir_size = 3
    xs = [
        2 * torch.rand(shape) - 1 for i in range(n)]

    reservoir = (
        pe.training.extensions.variable_statistics_plot.Reservoir(
            size=reservoir_size, data_shape=shape))
    for x in xs:
        reservoir.add(x)
    idxs, data = reservoir.get_data()

    assert len(idxs) == reservoir_size
    assert len(data) == reservoir_size
    assert idxs.ndim == 1
    assert data[0].shape == xs[0].shape
    numpy.testing.assert_almost_equal(idxs, numpy.sort(idxs))


def test_statistician_percentile():
    shape = (2, 7, 3)
    x = 2 * torch.rand(shape) - 1

    percentile_sigmas = (0., 100.)  # min, max
    statistician = (
        pe.training.extensions.variable_statistics_plot.Statistician(
                collect_mean=True, collect_std=True,
                percentile_sigmas=percentile_sigmas))
    stat = statistician(x, axis=None, dtype=x.dtype)

    for s in six.itervalues(stat):
        assert s.dtype == x.dtype

    assert torch.allclose(stat['mean'], torch.mean(x))
    assert torch.allclose(stat['std'], torch.std(x))

    percentile = stat['percentile']
    assert len(percentile) == 2

    assert torch.allclose(percentile[0], torch.min(x))
    assert torch.allclose(percentile[1], torch.max(x))
