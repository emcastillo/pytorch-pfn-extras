import collections
import contextlib
import os
import time

from pytorch_extensions.training import extension as extension_module
from pytorch_extensions.training import trigger as trigger_module
from pytorch_extensions.reporter import Reporter


# Select the best-resolution timer function
try:
    _get_time = time.perf_counter
except AttributeError:
    if os.name == 'nt':
        _get_time = time.clock
    else:
        _get_time = time.time


class FoolUpdater:

    def __init__(self, start_iteration, iters_per_epoch):
        self.iteration = start_iteration
        self._iters_per_epoch = iters_per_epoch

    @property
    def epoch(self):
        return self.iteration // self._iters_per_epoch

    @property
    def epoch_detail(self):
        return self.iteration / self._iters_per_epoch


class _ExtensionEntry:

    def __init__(self, extension, priority, trigger, call_before_training):
        self.extension = extension
        self.trigger = trigger
        self.priority = priority
        self.call_before_training = call_before_training

    def state_dict(self):
        state = {}
        if hasattr(self.extension, 'state_dict'):
            state['extension'] = self.extension.state_dict()
        if hasattr(self.trigger, 'state_dict'):
            state['trigger'] = self.trigger.state_dict()
        return state

    def load_state_dict(self, to_load):
        if 'extension' in to_load:
            self.extension.load_state_dict(to_load['extension'])
        if 'trigger' in to_load:
            self.trigger.load_state_dict(to_load['trigger'])


class _BaseExtensionsManager:
    """
    Keeps track of the extensions and the current status
    """

    updater = None

    def __init__(
            self,
            models,
            optimizers,
            max_epochs,
            extensions,
            out_dir='result'):
        if extensions is None:
            extensions = []
        self.stop_trigger = trigger_module.get_trigger((max_epochs, 'epoch'))
        self.observation = {}
        self.out = out_dir
        if not os.path.exists(self.out):
            os.makedirs(self.out)
        self.reporter = Reporter()

        for name in models:
            model = models[name]
            self.reporter.add_observer(name, model)
            self.reporter.add_observers(
                name, model.named_modules())

        self._models = models
        self._optimizers = optimizers
        self.max_epochs = max_epochs
        self._start_iteration = 0
        # Defer!
        self._start_time = None
        self._extensions = collections.OrderedDict()
        for ext in extensions:
            self.extend(ext)

    @property
    def elapsed_time(self):
        return _get_time()-self._start_time

    @property
    def is_before_training(self):
        return self.updater is None or self.updater.iteration == 0

    def _prepare_for_training(self, start_iteration, iters_per_epoch):
        assert self.updater is None
        self.updater = FoolUpdater(start_iteration, iters_per_epoch)

    def start_extensions(self):
        exts = self._extensions
        extension_order = sorted(
            exts.keys(),
            key=lambda name: exts[name].priority, reverse=True)
        self.extensions = [(name, exts[name])
                           for name in extension_order]

        # invoke initializer of each extension
        for _, entry in self.extensions:
            initializer = getattr(entry.extension, 'initialize', None)
            finished = getattr(entry.trigger, 'finished', False)
            if initializer and not finished:
                initializer(self)

        # call extensions before training loop
        self.observation = {}
        with self.reporter.scope(self.observation):
            for name, entry in self.extensions:
                if entry.call_before_training:
                    entry.extension(self)

    def extend(self, extension, name=None, trigger=None, priority=None,
               *, call_before_training=False, **kwargs):
        """Registers an extension to the manager.

        :class:`Extension` is a callable object which is called after each
        update unless the corresponding trigger object decides to skip the
        iteration. The order of execution is determined by priorities:
        extensions with higher priorities are called earlier in each iteration.
        Extensions with the same priority are invoked in the order of
        registrations.

        If two or more extensions with the same name are registered, suffixes
        are added to the names of the second to last extensions. The suffix is
        ``_N`` where N is the ordinal of the extensions.

        See :class:`Extension` for the interface of extensions.

        Args:
            extension: Extension to register.
            name (str): Name of the extension. If it is omitted, the
                :attr:`Extension.name` attribute of the extension is used or
                the :attr:`Extension.default_name` attribute of the extension
                if `name` is is set to `None` or is undefined.
                Note that the name would be suffixed by an ordinal in case of
                duplicated names as explained above.
            trigger (tuple or Trigger): Trigger object that determines when to
                invoke the extension. If it is ``None``, ``extension.trigger``
                is used instead. If it is ``None`` and the extension does not
                have the trigger attribute, the extension is triggered at every
                iteration by default. If the trigger is not callable, it is
                passed to :class:`IntervalTrigger` to build an interval
                trigger.
            call_before_training (bool): Flag to call extension before
                training. Default is ``False``.
            priority (int): Invocation priority of the extension. Extensions
                are invoked in the descending order of priorities in each
                iteration. If this is ``None``, ``extension.priority`` is used
                instead.

        """
        if name is None:
            name = getattr(extension, 'name', None)
            if name is None:
                name = getattr(extension, 'default_name', None)
                if name is None:
                    name = getattr(extension, '__name__', None)
                    if name is None:
                        raise TypeError('name is not given for the extension')
        if name == 'training':
            raise ValueError(
                'the name "training" is prohibited as an extension name')

        if trigger is None:
            trigger = getattr(extension, 'trigger', (1, 'iteration'))
        trigger = trigger_module.get_trigger(trigger)

        if priority is None:
            priority = getattr(
                extension, 'priority', extension_module.PRIORITY_READER)

        modified_name = name
        ordinal = 0
        while modified_name in self._extensions:
            ordinal += 1
            modified_name = '%s_%d' % (name, ordinal)

        extension.name = modified_name
        self._extensions[modified_name] = _ExtensionEntry(
            extension, priority, trigger, call_before_training)

    def get_extension(self, name):
        """Returns the extension of a given name.

        Args:
            name (str): Name of the extension.

        Returns:
            Extension.

        """
        extensions = self._extensions
        if name in extensions:
            return extensions[name].extension
        else:
            raise ValueError('extension %s not found' % name)

    def run_extensions(self):
        for name, entry in self.extensions:
            if entry.trigger(self):
                entry.extension(self)

    def state_dict(self):
        to_save = {}
        if self.updater is not None:
            to_save['_start_iteration'] = self.updater.iteration
        else:
            to_save['_start_iteration'] = 0
        # Save manager status ?
        to_save['models'] = {name: self._models[name].state_dict()
                             for name in self._models}
        to_save['optimizers'] = {name: self._optimizers[name].state_dict()
                                 for name in self._optimizers}
        to_save['extensions'] = {name: self._extensions[name].state_dict()
                                 for name in self._extensions}
        return to_save

    def load_state_dict(self, to_load):
        self._start_iteration = to_load['_start_iteration']
        if self.updater is not None:
            self.updater.iteration = self._start_iteration
        for name in self._models:
            self._models[name].load_state_dict(to_load['models'][name])

        for name in self._optimizers:
            self._optimizers[name].load_state_dict(to_load['optimizers'][name])

        for name in self._extensions:
            self._extensions[name].load_state_dict(
                to_load['extensions'][name])


class ExtensionsManager(_BaseExtensionsManager):
    """
    Keeps track of the extensions and the current status
    """

    def __init__(
            self,
            models,
            optimizers,
            max_epochs,
            *,
            extensions=None,
            iters_per_epoch,
            out_dir='result'):
        super().__init__(
            models, optimizers, max_epochs, extensions, out_dir)
        if not (isinstance(iters_per_epoch, int) and iters_per_epoch >= 1):
            raise ValueError(
                'iters_per_epoch must be an integer >= 1 ({} given)'.format(
                    iters_per_epoch))
        self._prepare_for_training(0, iters_per_epoch)

    @contextlib.contextmanager
    def run_iteration(self):
        assert self.updater is not None
        if self._start_time is None:
            self._start_time = _get_time()
            self.start_extensions()

        self.observation = {}
        with self.reporter.scope(self.observation):
            try:
                yield
            finally:
                # In chainer, the iteration count was increased
                # just before calling the extensions, we need
                # to keep the semantics
                self.updater.iteration += 1
                self.run_extensions()


class IgniteExtensionsManager(_BaseExtensionsManager):
    def __init__(
            self,
            engine,
            models,
            optimizers,
            max_epochs,
            *,
            extensions=None,
            out_dir='result'):
        super().__init__(models, optimizers, max_epochs, extensions, out_dir)
        self.engine = engine
        self.set_ignite_handlers()

    def set_ignite_handlers(self):
        from ignite.engine import Events
        # Set a handler that sets the reporter scope on every iteration
        @self.engine.on(Events.ITERATION_STARTED)
        def set_reporter_on_iter(engine):
            self.observation = {}
            self.cm = self.reporter.scope(self.observation)
            self.cm.__enter__()

        @self.engine.on(Events.STARTED)
        def set_training_started(engine):
            start_iteration = self._start_iteration
            self.engine.state.iteration = start_iteration
            self._start_time = _get_time()
            iters_per_epoch = len(engine.state.dataloader)
            self._prepare_for_training(start_iteration, iters_per_epoch)
            self.start_extensions()
            # Make all the next
            # handlers to be executed after user defined ones
            @self.engine.on(Events.ITERATION_COMPLETED)
            def run_extensions_on_iter(engine):
                self.updater.iteration = engine.state.iteration
                self.run_extensions()

            # This should be the last extension to be run
            @self.engine.on(Events.ITERATION_COMPLETED)
            def close_reporter_on_iter(engine):
                self.cm.__exit__(None, None, None)

    def state_dict(self):
        to_save = super().state_dict()
        to_save['_start_epoch'] = self.engine.state.epoch
        to_save['_start_iteration'] = self.engine.state.iteration
        return to_save

    def load_state_dict(self, to_load):
        super().load_state_dict(to_load)
        self.engine.state.epoch = self._start_epoch
        self.engine.state.iteration = self._start_iteration
