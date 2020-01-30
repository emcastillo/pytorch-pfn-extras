import pytest

from torch import nn

from pytorch_extensions import training


def test_fool_updater():
    updater = training.manager.FoolUpdater(9, 4)
    assert updater.iteration == 9
    assert updater.epoch_size == 4
    assert updater.epoch == 2
    assert updater.epoch_detail == 2.25

    updater.iteration = 15
    assert updater.epoch == 3
    assert updater.epoch_detail == 3.75


class _DummyExtension(object):

    def __init__(self, extension_id, call_record, init_record):
        self.extension_id = extension_id
        self.call_record = call_record
        self.init_record = init_record

    def __call__(self, manager):
        self.call_record.append(self.extension_id)


class _DummyExtensionInitialize(_DummyExtension):

    def initialize(self, manager):
        self.init_record.append(self.extension_id)


def test_extensions_manager_extensions():
    model = nn.Module()
    optimizer = object()
    max_epochs = 5
    epoch_size = 4
    manager = training.ExtensionsManager(
        {'model_name': model},
        {'optimizer_name': optimizer},
        max_epochs,
        []
    )

    call_record = []
    init_record = []

    exts = [
        _DummyExtension(0, call_record, init_record),
        _DummyExtensionInitialize(1, call_record, init_record),
        _DummyExtension(2, call_record, init_record),
        _DummyExtensionInitialize(3, call_record, init_record),
        _DummyExtensionInitialize(4, call_record, init_record),
    ]

    manager.extend(exts[0], 'ext0', priority=2, call_before_training=True)
    manager.extend(exts[1], 'ext1', priority=1, call_before_training=False)
    manager.extend(exts[2], 'ext2', priority=3, call_before_training=False)
    manager.extend(exts[3], 'ext3', priority=0, call_before_training=True)
    manager.extend(exts[4], 'ext4', priority=4, call_before_training=True)

    assert manager.get_extension('ext0') is exts[0]
    assert manager.get_extension('ext1') is exts[1]
    assert manager.get_extension('ext2') is exts[2]
    assert manager.get_extension('ext3') is exts[3]
    assert manager.get_extension('ext4') is exts[4]

    with pytest.raises(ValueError):
        manager.get_extension('ext10')

    for it in range(max_epochs * epoch_size):
        call_record.clear()
        init_record.clear()

        with manager.run_iteration(iteration=it, epoch_size=epoch_size):
            assert manager.updater.iteration == it
            assert manager.updater.epoch_size == epoch_size

            if it == 0:
                assert call_record == [4, 0, 3]
                assert init_record == [4, 1, 3]
            else:
                assert call_record == []
                assert init_record == []

            call_record.clear()
            init_record.clear()

        assert call_record == [4, 2, 0, 1, 3]
        assert init_record == []


class _StateDictObj():
    def __init__(self, *, state_dict=None, state_dict_to_be_loaded=None):
        super().__init__()
        self.called_load_state_dict = 0
        self._state_dict = state_dict
        self._state_dict_to_be_loaded = state_dict_to_be_loaded

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self.called_load_state_dict += 1
        assert state_dict is self._state_dict_to_be_loaded


class _StateDictModel(_StateDictObj, nn.Module):

    def forward(self, *args):
        pass


class _StateDictExtension(_StateDictObj):

    def __call__(self, manager):
        pass


def test_extensions_manager_state_dict():
    model_state_dict = object()
    optimizer_state_dict = object()
    extension_state_dict = object()
    max_epochs = 5
    epoch_size = 4
    passed_iteration = 11
    manager = training.ExtensionsManager(
        {'model_name': _StateDictModel(state_dict=model_state_dict)},
        {'optimizer_name': _StateDictObj(state_dict=optimizer_state_dict)},
        max_epochs,
        []
    )
    manager.extend(
        _StateDictExtension(
            state_dict=extension_state_dict), name='extension_name')

    for it in range(passed_iteration):
        with manager.run_iteration(iteration=it, epoch_size=epoch_size):
            pass

    state_dict = manager.state_dict()

    assert state_dict == {
        '_start_iteration': passed_iteration,
        'models': {'model_name': model_state_dict},
        'optimizers': {'optimizer_name': optimizer_state_dict},
        'extensions': {'extension_name': {
            'extension': extension_state_dict,
            'trigger': {
                '_previous_iteration': passed_iteration,
                '_previous_epoch_detail': passed_iteration / epoch_size
            },
        }},
    }

    new_model = _StateDictModel(state_dict_to_be_loaded=model_state_dict)
    new_optimizer = _StateDictObj(state_dict_to_be_loaded=optimizer_state_dict)
    new_extension = _StateDictExtension(
        state_dict_to_be_loaded=extension_state_dict)
    new_manager = training.ExtensionsManager(
        {'model_name': new_model},
        {'optimizer_name': new_optimizer},
        max_epochs,
        []
    )
    new_manager.extend(new_extension, name='extension_name')
    new_manager.load_state_dict(state_dict)
    assert new_model.called_load_state_dict == 1
    assert new_optimizer.called_load_state_dict == 1
    assert new_optimizer.called_load_state_dict == 1
