import os
import unittest

import numpy
import torch
import six

import pytorch_extensions as pe

try:
    import matplotlib
    _available = True
except ImportError:
    _available = False


class TestVariableStatisticsPlot(unittest.TestCase):

    def setUp(self):
        self.filename = 'variable_statistics_plot_test.png'
        self.iterations = 2
        extension_trigger = (1, 'iteration')
        self.manager = pe.training.ExtensionsManager({}, [], 2, [])

        x = torch.rand(1, 2, 3)
        self.extension = pe.training.extensions.VariableStatisticsPlot(
            x, trigger=extension_trigger, filename=self.filename)
        self.manager.extend(self.extension, extension_trigger)

    # In the following we explicitly use plot_report._available instead of
    # PlotReport.available() because in some cases `test_available()` fails
    # because it sometimes does not raise UserWarning despite
    # matplotlib is not installed (this is due to the difference between
    # the behavior of unittest in python2 and that in python3).
    @unittest.skipUnless(_available, 'matplotlib is not installed')
    def test_run_and_save_plot(self):
        matplotlib.use('Agg')
        try:
            for i in range(self.iterations):
                cur_it = 1+i
                with self.manager.run_iteration(
                        epoch=1, iteration=cur_it, epoch_size=2):
                    pass
        finally:
            os.remove(os.path.join(self.manager.out, self.filename))


class TestReservoir(unittest.TestCase):

    def setUp(self):
        self.shape = (2, 7, 3)
        self.n = 5
        self.reservoir_size = 3
        self.xs = [
            2 * torch.rand(self.shape) - 1 for i in range(self.n)]

    def test_reservoir_size(self):
        self.reservoir = (
            pe.training.extensions.variable_statistics_plot.Reservoir(
                size=self.reservoir_size, data_shape=self.shape))
        for x in self.xs:
            self.reservoir.add(x)
        idxs, data = self.reservoir.get_data()

        assert len(idxs) == self.reservoir_size
        assert len(data) == self.reservoir_size
        assert idxs.ndim == 1
        assert data[0].shape == self.xs[0].shape
        numpy.testing.assert_almost_equal(idxs, numpy.sort(idxs))


class TestStatistician(unittest.TestCase):

    def setUp(self):
        self.shape = (2, 7, 3)
        self.x = 2 * torch.rand(self.shape) - 1

    def test_statistician_percentile(self):
        self.percentile_sigmas = (0., 100.)  # min, max
        self.statistician = (
            pe.training.extensions.variable_statistics_plot.Statistician(
                    collect_mean=True, collect_std=True,
                    percentile_sigmas=self.percentile_sigmas))
        stat = self.statistician(self.x, axis=None, dtype=self.x.dtype)

        for s in six.itervalues(stat):
            assert s.dtype == self.x.dtype

        assert torch.allclose(stat['mean'], torch.mean(self.x))
        assert torch.allclose(stat['std'], torch.std(self.x))

        percentile = stat['percentile']
        assert len(percentile) == 2

        assert torch.allclose(percentile[0], torch.min(self.x))
        assert torch.allclose(percentile[1], torch.max(self.x))
