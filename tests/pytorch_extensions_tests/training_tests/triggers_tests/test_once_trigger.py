import random
import pytest

import pytorch_extensions as pte


_parametrize = pytest.mark.parametrize(
    'iter_per_epoch,call_on_resume,resume',
    [
        # basic
        (5, False, 4),
        # call on resume
        (5, True, 4),
        # unaligned epoch
        (2.5, False, 3),
        # unaligned epoch, call on resume
        (2.5, True, 3),
        # tiny epoch
        (0.5, False, 4),
        # tiny epoch, call on resume
        (0.5, True, 4),
    ])


@_parametrize
def test_trigger(iter_per_epoch, call_on_resume, resume):
    del resume  # resume is ignored
    expected = [True] + [False] * 6
    finished = [False] + [True] * 6
    manager = pte.training.ExtensionsManager({}, [], 100, [])
    trigger = pte.training.triggers.OnceTrigger(call_on_resume)
    for it, (e, f) in enumerate(zip(expected, finished)):
        assert trigger.finished == f
        assert trigger(manager) == e
        with manager.run_iteration(iteration=it, epoch_size=iter_per_epoch):
            pass


@_parametrize
def test_resumed_trigger(iter_per_epoch, call_on_resume, resume):
    expected = [True] + [False] * 6
    finished = [False] + [True] * 6
    if call_on_resume:
        expected[resume] = True
        finished[resume] = False
    manager = pte.training.ExtensionsManager({}, [], 100, [])
    trigger = pte.training.triggers.OnceTrigger(call_on_resume)
    for it, (e, f) in enumerate(zip(expected[:resume], finished[:resume])):
        with manager.run_iteration(iteration=it, epoch_size=iter_per_epoch):
            pass
        assert trigger.finished == f
        assert trigger(manager) == e
    state = trigger.state_dict()

    trigger2 = pte.training.triggers.OnceTrigger(call_on_resume)
    trigger2.load_state_dict(state)
    for it, (e, f) in enumerate(zip(expected[resume:], finished[resume:])):
        with manager.run_iteration(iteration=it, epoch_size=iter_per_epoch):
            pass
        assert trigger2.finished == f
        assert trigger2(manager) == e


@_parametrize
def test_trigger_sparse_call(iter_per_epoch, call_on_resume, resume):
    del resume  # resume is ignored
    expected = [True] + [False] * 6
    finished = [False] + [True] * 6
    for _ in range(10):
        manager = pte.training.ExtensionsManager({}, [], 100, [])
        trigger = pte.training.triggers.OnceTrigger(call_on_resume)
        accumulated = False
        accumulated_finished = True
        for it, (e, f) in enumerate(zip(expected, finished)):
            with manager.run_iteration(
                    iteration=it, epoch_size=iter_per_epoch):
                accumulated = accumulated or e
                accumulated_finished = accumulated_finished and f
                if random.randrange(2):
                    assert trigger.finished == accumulated_finished
                    assert trigger(manager) == accumulated
                    accumulated = False
                    accumulated_finished = True
