import unittest

import chainer.testing
import numpy
import torch

import pytorch_extensions as pte


class DummyModel(torch.nn.Module):

    def __init__(self, test):
        super(DummyModel, self).__init__()
        self.args = []
        self.test = test

    def forward(self, x):
        self.args.append(x)
        pte.reporter.report({'loss': x.sum()}, self)


class DummyModelTwoArgs(torch.nn.Module):

    def __init__(self, test):
        super(DummyModelTwoArgs, self).__init__()
        self.args = []
        self.test = test

    def forward(self, x, y):
        self.args.append((x, y))
        pte.reporter.report({'loss': x.sum() + y.sum()}, self)


class DummyConverter(object):

    def __init__(self, return_values):
        self.args = []
        self.iterator = iter(return_values)

    def __call__(self, batch, device):
        self.args.append({'batch': batch, 'device': device})
        return next(self.iterator)


def _torch_batch_to_numpy(batch):
    # In Pytorch, a batch has the batch dimension. Squeeze it for comparison.
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] == 1
    return batch.squeeze(0).numpy()


class TestEvaluator(unittest.TestCase):

    def setUp(self):
        self.data = [
            numpy.random.uniform(-1, 1, (3, 4)).astype('f') for _ in range(2)]
        self.batches = [
            numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')
            for _ in range(2)]

        self.data_loader = torch.utils.data.DataLoader(self.data)
        self.converter = DummyConverter(self.batches)
        self.target = DummyModel(self)
        self.evaluator = pte.training.extensions.Evaluator(
            self.data_loader, self.target, converter=self.converter)
        self.expect_mean = numpy.mean([numpy.sum(x) for x in self.batches])

    def test_evaluate(self):
        reporter = pte.reporter.Reporter()
        reporter.add_observer('target', self.target)
        with reporter:
            mean = self.evaluator.evaluate()

        # No observation is reported to the current reporter. Instead the
        # evaluator collect results in order to calculate their mean.
        self.assertEqual(len(reporter.observation), 0)

        # The converter gets results of the data loader.
        self.assertEqual(len(self.converter.args), len(self.data))
        for i in range(len(self.data)):
            numpy.testing.assert_array_equal(
                _torch_batch_to_numpy(self.converter.args[i]['batch']),
                self.data[i])
            self.assertIsNone(self.converter.args[i]['device'])

        # The model gets results of converter.
        self.assertEqual(len(self.target.args), len(self.batches))
        for i in range(len(self.batches)):
            numpy.testing.assert_array_equal(
                self.target.args[i], self.batches[i])

        self.assertAlmostEqual(mean['target/loss'], self.expect_mean, places=4)

        self.evaluator.finalize()

    def test_call(self):
        mean = self.evaluator()
        # 'main' is used by default
        self.assertAlmostEqual(mean['main/loss'], self.expect_mean, places=4)

    def test_evaluator_name(self):
        self.evaluator.name = 'eval'
        mean = self.evaluator()
        # name is used as a prefix
        self.assertAlmostEqual(
            mean['eval/main/loss'], self.expect_mean, places=4)

    def test_current_report(self):
        reporter = pte.reporter.Reporter()
        with reporter:
            mean = self.evaluator()
        # The result is reported to the current reporter.
        self.assertEqual(reporter.observation, mean)


@chainer.testing.parameterize(
    {'device': None},
    {'device': 'cpu'},
    {'device': 'cuda'},
)
class TestEvaluatorTupleData(unittest.TestCase):

    def setUp(self):
        self.data = [
            numpy.random.uniform(-1, 1, (3, 4)).astype('f') for _ in range(2)]
        self.batches = [
            (numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'),
             numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'))
            for _ in range(2)]

    def prepare(self, data, batches, device):
        data_loader = torch.utils.data.DataLoader(data)
        converter = DummyConverter(batches)
        target = DummyModelTwoArgs(self)
        evaluator = pte.training.extensions.Evaluator(
            data_loader, target, converter=converter, device=device)
        return data_loader, converter, target, evaluator

    def test_evaluate(self):
        data = self.data
        batches = self.batches
        device = None if self.device is None else torch.device(self.device)

        data_loader, converter, target, evaluator = (
            self.prepare(data, batches, device))

        reporter = pte.reporter.Reporter()
        reporter.add_observer('target', target)
        with reporter:
            mean = evaluator.evaluate()

        # The converter gets results of the data loader and the device number.
        self.assertEqual(len(converter.args), len(data))
        expected_device_arg = device

        for i in range(len(data)):
            numpy.testing.assert_array_equal(
                _torch_batch_to_numpy(converter.args[i]['batch']),
                self.data[i])
            self.assertEqual(converter.args[i]['device'], expected_device_arg)

        # The model gets results of converter.
        self.assertEqual(len(target.args), len(batches))
        for i in range(len(batches)):
            numpy.testing.assert_array_equal(
                target.args[i], self.batches[i])

        expect_mean = numpy.mean([numpy.sum(x) for x in self.batches])
        self.assertAlmostEqual(
            mean['target/loss'], expect_mean, places=4)


class TestEvaluatorDictData(unittest.TestCase):

    def setUp(self):
        self.data = range(2)
        self.batches = [
            {'x': numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'),
             'y': numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')}
            for _ in range(2)]

        self.data_loader = torch.utils.data.DataLoader(self.data)
        self.converter = DummyConverter(self.batches)
        self.target = DummyModelTwoArgs(self)
        self.evaluator = pte.training.extensions.Evaluator(
            self.data_loader, self.target, converter=self.converter)

    def test_evaluate(self):
        reporter = pte.reporter.Reporter()
        reporter.add_observer('target', self.target)
        with reporter:
            mean = self.evaluator.evaluate()

        # The model gets results of converter.
        self.assertEqual(len(self.target.args), len(self.batches))
        for i in range(len(self.batches)):
            numpy.testing.assert_array_equal(
                self.target.args[i][0], self.batches[i]['x'])
            numpy.testing.assert_array_equal(
                self.target.args[i][1], self.batches[i]['y'])

        expect_mean = numpy.mean(
            [numpy.sum(x['x']) + numpy.sum(x['y']) for x in self.batches])
        self.assertAlmostEqual(mean['target/loss'], expect_mean, places=4)


class TestEvaluatorWithEvalFunc(unittest.TestCase):

    def setUp(self):
        self.data = [
            numpy.random.uniform(-1, 1, (3, 4)).astype('f') for _ in range(2)]
        self.batches = [
            numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')
            for _ in range(2)]

        self.data_loader = torch.utils.data.DataLoader(self.data)
        self.converter = DummyConverter(self.batches)
        self.target = DummyModel(self)
        self.evaluator = pte.training.extensions.Evaluator(
            self.data_loader, {}, converter=self.converter,
            eval_func=self.target)

    def test_evaluate(self):
        reporter = pte.reporter.Reporter()
        reporter.add_observer('target', self.target)
        with reporter:
            self.evaluator.evaluate()

        # The model gets results of converter.
        self.assertEqual(len(self.target.args), len(self.batches))
        for i in range(len(self.batches)):
            numpy.testing.assert_array_equal(
                self.target.args[i], self.batches[i])


class TestEvaluatorProgressBar(unittest.TestCase):

    def setUp(self):
        self.data = [
            numpy.random.uniform(-1, 1, (3, 4)).astype('f') for _ in range(2)]

        self.data_loader = torch.utils.data.DataLoader(self.data, batch_size=1)
        self.target = DummyModel(self)
        self.evaluator = pte.training.extensions.Evaluator(
            self.data_loader, {}, eval_func=self.target, progress_bar=True)

    def test_evaluator(self):
        reporter = pte.reporter.Reporter()
        reporter.add_observer('target', self.target)
        with reporter:
            self.evaluator.evaluate()
