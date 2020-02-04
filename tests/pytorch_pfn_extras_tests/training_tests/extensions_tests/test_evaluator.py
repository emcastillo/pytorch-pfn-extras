import numpy
import pytest
import torch

import pytorch_pfn_extras as ppe


class DummyModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.args = []

    def forward(self, x):
        self.args.append(x)
        ppe.reporter.report({'loss': x.sum()}, self)


class DummyModelTwoArgs(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.args = []

    def forward(self, x, y):
        self.args.append((x, y))
        ppe.reporter.report({'loss': x.sum() + y.sum()}, self)


def _torch_batch_to_numpy(batch):
    # In Pytorch, a batch has the batch dimension. Squeeze it for comparison.
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] == 1
    return batch.squeeze(0).numpy()


@pytest.fixture(scope='function')
def evaluator_dummies():
    data = [
        numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')
        for _ in range(2)]

    data_loader = torch.utils.data.DataLoader(data)
    target = DummyModel()
    evaluator = ppe.training.extensions.Evaluator(data_loader, target)
    expect_mean = numpy.mean([numpy.sum(x) for x in data])
    return data, data_loader, target, evaluator, expect_mean


def test_evaluate(evaluator_dummies):
    data, data_loader, target, evaluator, expect_mean = evaluator_dummies

    reporter = ppe.reporter.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        mean = evaluator.evaluate()

    # No observation is reported to the current reporter. Instead the
    # evaluator collect results in order to calculate their mean.
    assert len(reporter.observation) == 0

    assert len(target.args) == len(data)
    for i in range(len(data)):
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i]), data[i])

    numpy.testing.assert_almost_equal(
        mean['target/loss'], expect_mean, decimal=4)

    evaluator.finalize()


def test_call(evaluator_dummies):
    data, data_loader, target, evaluator, expect_mean = evaluator_dummies

    mean = evaluator()
    # 'main' is used by default
    numpy.testing.assert_almost_equal(
        mean['main/loss'], expect_mean, decimal=4)


def test_evaluator_name(evaluator_dummies):
    data, data_loader, target, evaluator, expect_mean = evaluator_dummies

    evaluator.name = 'eval'
    mean = evaluator()
    # name is used as a prefix
    numpy.testing.assert_almost_equal(
        mean['eval/main/loss'], expect_mean, decimal=4)


def test_current_report(evaluator_dummies):
    data, data_loader, target, evaluator, expect_mean = evaluator_dummies

    reporter = ppe.reporter.Reporter()
    with reporter:
        mean = evaluator()
    # The result is reported to the current reporter.
    assert reporter.observation == mean


def test_evaluator_tuple_data():
    data = [
        (numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'),
         numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'))
        for _ in range(2)]

    data_loader = torch.utils.data.DataLoader(data)
    target = DummyModelTwoArgs()
    evaluator = ppe.training.extensions.Evaluator(data_loader, target)

    reporter = ppe.reporter.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        mean = evaluator.evaluate()

    assert len(target.args) == len(data)
    for i in range(len(data)):
        assert len(target.args[i]) == len(data[i])
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i][0]), data[i][0])
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i][1]), data[i][1])

    expect_mean = numpy.mean([numpy.sum(x) for x in data])
    numpy.testing.assert_almost_equal(
        mean['target/loss'], expect_mean, decimal=4)


def test_evaluator_dict_data():
    data = [
        {'x': numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'),
         'y': numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')}
        for _ in range(2)]

    data_loader = torch.utils.data.DataLoader(data)
    target = DummyModelTwoArgs()
    evaluator = ppe.training.extensions.Evaluator(data_loader, target)

    reporter = ppe.reporter.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        mean = evaluator.evaluate()

    assert len(target.args) == len(data)
    for i in range(len(data)):
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i][0]), data[i]['x'])
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i][1]), data[i]['y'])

    expect_mean = numpy.mean(
        [numpy.sum(x['x']) + numpy.sum(x['y']) for x in data])
    numpy.testing.assert_almost_equal(
        mean['target/loss'], expect_mean, decimal=4)


def test_evaluator_with_eval_func():
    data = [
        numpy.random.uniform(-1, 1, (3, 4)).astype('f') for _ in range(2)]

    data_loader = torch.utils.data.DataLoader(data)
    target = DummyModel()
    evaluator = ppe.training.extensions.Evaluator(
        data_loader, {}, eval_func=target)

    reporter = ppe.reporter.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        evaluator.evaluate()

    assert len(target.args) == len(data)
    for i in range(len(data)):
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i]), data[i])


def test_evaluator_progress_bar():
    data = [
        numpy.random.uniform(-1, 1, (3, 4)).astype('f') for _ in range(2)]

    data_loader = torch.utils.data.DataLoader(data, batch_size=1)
    target = DummyModel()
    evaluator = ppe.training.extensions.Evaluator(
        data_loader, {}, eval_func=target, progress_bar=True)

    reporter = ppe.reporter.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        evaluator.evaluate()
