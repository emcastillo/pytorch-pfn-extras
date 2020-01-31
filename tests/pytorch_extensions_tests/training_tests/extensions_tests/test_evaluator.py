import numpy
import pytest
import torch

import pytorch_extensions as pte


class DummyModel(torch.nn.Module):

    def __init__(self):
        super(DummyModel, self).__init__()
        self.args = []

    def forward(self, x):
        self.args.append(x)
        pte.reporter.report({'loss': x.sum()}, self)


class DummyModelTwoArgs(torch.nn.Module):

    def __init__(self):
        super(DummyModelTwoArgs, self).__init__()
        self.args = []

    def forward(self, x, y):
        self.args.append((x, y))
        pte.reporter.report({'loss': x.sum() + y.sum()}, self)


def _torch_batch_to_numpy(batch):
    # In Pytorch, a batch has the batch dimension. Squeeze it for comparison.
    assert isinstance(batch, torch.Tensor)
    assert batch.shape[0] == 1
    return batch.squeeze(0).numpy()


@pytest.fixture(scope='function')
def evaluator_dummies():
    batches = [
        numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')
        for _ in range(2)]

    data_loader = torch.utils.data.DataLoader(batches)
    target = DummyModel()
    evaluator = pte.training.extensions.Evaluator(data_loader, target)
    expect_mean = numpy.mean([numpy.sum(x) for x in batches])
    return batches, data_loader, target, evaluator, expect_mean


def test_evaluate(evaluator_dummies):
    batches, data_loader, target, evaluator, expect_mean = evaluator_dummies

    reporter = pte.reporter.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        mean = evaluator.evaluate()

    # No observation is reported to the current reporter. Instead the
    # evaluator collect results in order to calculate their mean.
    assert len(reporter.observation) == 0

    assert len(target.args) == len(batches)
    for i in range(len(batches)):
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i]), batches[i])

    numpy.testing.assert_almost_equal(
        mean['target/loss'], expect_mean, decimal=4)

    evaluator.finalize()


def test_call(evaluator_dummies):
    batches, data_loader, target, evaluator, expect_mean = evaluator_dummies

    mean = evaluator()
    # 'main' is used by default
    numpy.testing.assert_almost_equal(
        mean['main/loss'], expect_mean, decimal=4)


def test_evaluator_name(evaluator_dummies):
    batches, data_loader, target, evaluator, expect_mean = evaluator_dummies

    evaluator.name = 'eval'
    mean = evaluator()
    # name is used as a prefix
    numpy.testing.assert_almost_equal(
        mean['eval/main/loss'], expect_mean, decimal=4)


def test_current_report(evaluator_dummies):
    batches, data_loader, target, evaluator, expect_mean = evaluator_dummies

    reporter = pte.reporter.Reporter()
    with reporter:
        mean = evaluator()
    # The result is reported to the current reporter.
    assert reporter.observation == mean


@pytest.mark.parametrize('device', [None, 'cpu', 'cuda'])
def test_evaluator_tuple_data(device):
    batches = [
        (numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'),
         numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'))
        for _ in range(2)]

    device = None if device is None else torch.device(device)
    data_loader = torch.utils.data.DataLoader(batches)
    target = DummyModelTwoArgs()
    evaluator = pte.training.extensions.Evaluator(
        data_loader, target, device=device)

    reporter = pte.reporter.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        mean = evaluator.evaluate()

    assert len(target.args) == len(batches)
    for i in range(len(batches)):
        assert len(target.args[i]) == len(batches[i])
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i][0]), batches[i][0])
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i][1]), batches[i][1])

    expect_mean = numpy.mean([numpy.sum(x) for x in batches])
    numpy.testing.assert_almost_equal(
        mean['target/loss'], expect_mean, decimal=4)


def test_evaluator_dict_data():
    batches = [
        {'x': numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f'),
         'y': numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')}
        for _ in range(2)]

    data_loader = torch.utils.data.DataLoader(batches)
    target = DummyModelTwoArgs()
    evaluator = pte.training.extensions.Evaluator(data_loader, target)

    reporter = pte.reporter.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        mean = evaluator.evaluate()

    assert len(target.args) == len(batches)
    for i in range(len(batches)):
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i][0]), batches[i]['x'])
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i][1]), batches[i]['y'])

    expect_mean = numpy.mean(
        [numpy.sum(x['x']) + numpy.sum(x['y']) for x in batches])
    numpy.testing.assert_almost_equal(
        mean['target/loss'], expect_mean, decimal=4)


def test_evaluator_with_eval_func():
    batches = [
        numpy.random.uniform(-1, 1, (3, 4)).astype('f') for _ in range(2)]

    data_loader = torch.utils.data.DataLoader(batches, batch_size=1)
    target = DummyModel()
    evaluator = pte.training.extensions.Evaluator(
        data_loader, {}, eval_func=target)

    reporter = pte.reporter.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        evaluator.evaluate()

    assert len(target.args) == len(batches)
    for i in range(len(batches)):
        numpy.testing.assert_array_equal(
            _torch_batch_to_numpy(target.args[i]), batches[i])


def test_evaluator_progress_bar():
    data = [
        numpy.random.uniform(-1, 1, (3, 4)).astype('f') for _ in range(2)]

    data_loader = torch.utils.data.DataLoader(data, batch_size=1)
    target = DummyModel()
    evaluator = pte.training.extensions.Evaluator(
        data_loader, {}, eval_func=target, progress_bar=True)

    reporter = pte.reporter.Reporter()
    reporter.add_observer('target', target)
    with reporter:
        evaluator.evaluate()
