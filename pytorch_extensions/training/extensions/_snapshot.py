import os

import torch

from pytorch_extensions import argument
from pytorch_extensions.training import extension
from pytorch_extensions.training.extensions import snapshot_writers


def _find_snapshot_files(fmt, path):
    '''Only prefix and suffix match

    TODO(kuenishi): currently clean format string such as
    "snapshot{.iteration}.npz" can only be parsed, but tricky (or
    invalid) formats like "snapshot{{.iteration}}.npz" are hard to
    detect and to properly show errors, just ignored or fails so far.

    Args:
        fmt (str): format string to match with file names of
            existing snapshots, where prefix and suffix are
            only examined. Also, files' staleness is judged
            by timestamps. The default is metime.
        path (str): a directory path to search for snapshot files.

    Returns:
        A sorted list of pair of ``mtime, filename``, whose file
        name that matched the format ``fmt`` directly under ``path``.

    '''
    prefix = fmt.split('{')[0]
    suffix = fmt.split('}')[-1]

    matched_files = (file for file in os.listdir(path)
                     if file.startswith(prefix) and file.endswith(suffix))

    def _prepend_mtime(f):
        t = os.stat(os.path.join(path, f)).st_mtime
        return (t, f)

    return sorted(_prepend_mtime(file) for file in matched_files)


def _find_latest_snapshot(fmt, path):
    """Finds the latest snapshots in a directory

    Args:
        fmt (str): format string to match with file names of
            existing snapshots, where prefix and suffix are
            only examined. Also, files' staleness is judged
            by timestamps. The default is metime.
        path (str): a directory path to search for snapshot files.

    Returns:
        Latest snapshot file, in terms of a file that has newest
        ``mtime`` that matches format ``fmt`` directly under
        ``path``. If no such file found, it returns ``None``.

    """
    snapshot_files = _find_snapshot_files(fmt, path)

    if len(snapshot_files) > 0:
        _, filename = snapshot_files[-1]
        return filename

    return None


def _find_stale_snapshots(fmt, path, num_retain):
    """Finds stale snapshots in a directory, retaining several files

    Args:
        fmt (str): format string to match with file names of
            existing snapshots, where prefix and suffix are
            only examined. Also, files' staleness is judged
            by timestamps. The default is metime.
        path (str): a directory path to search for snapshot files.
        num_retain (int): Number of snapshot files to retain
            through the cleanup. Must be a positive integer for any cleanup to
            take place.

    Returns:
        Generator that yields stale files that matches format
        ``fmt`` directly under ``path`` and with older ``mtime``,
        excluding newest ``num_retain`` files.

    """
    snapshot_files = _find_snapshot_files(fmt, path)
    num_remove = len(snapshot_files) - num_retain
    if num_remove > 0:
        for _, filename in snapshot_files[:num_remove]:
            yield filename
    return


def snapshot_object(target, filename, savefun=None, **kwargs):
    """snapshot_object(target, filename, savefun=None, \
*, condition=None, writer=None, snapshot_on_error=False, \
num_retain=-1, autoload=False)

    Returns a trainer extension to take snapshots of a given object.

    This extension serializes the given object and saves it to the output
    directory.

    This extension is called once per epoch by default. To take a
    snapshot at a different interval, a trigger object specifying the
    required interval can be passed along with this extension
    to the `extend()` method of the trainer.

    The default priority is -100, which is lower than that of most
    built-in extensions.

    Args:
        target: Object to serialize.
        filename (str): Name of the file into which the object is serialized.
            It can be a format string, where the trainer object is passed to
            the :meth:`str.format` method. For example,
            ``'snapshot_{.updater.iteration}'`` is converted to
            ``'snapshot_10000'`` at the 10,000th iteration.
        savefun: Function to save the object. It takes two arguments: the
            output file path and the object to serialize.
        condition: Condition object. It must be a callable object that returns
            boolean without any arguments. If it returns ``True``, the snapshot
            will be done.
            If not, it will be skipped. The default is a function that always
            returns ``True``.
        writer: Writer object.
            It must be a callable object.
            See below for the list of built-in writers.
            If ``savefun`` is other than ``None``, this argument must be
            ``None``. In that case, a
            :class:`~chainer.training.extensions.snapshot_writers.SimpleWriter`
            object instantiated with specified ``savefun`` argument will be
            used.
        snapshot_on_error (bool): Whether to take a snapshot in case trainer
            loop has been failed.
        num_retain (int): Number of snapshot files to retain
            through the cleanup. Must be a positive integer for any cleanup to
            take place. Automatic deletion of old snapshots only works when the
            filename is string.
        autoload (bool): With this enabled, the extension automatically
            finds the latest snapshot and loads the data to the target.
            Automatic loading only works when the filename is a string.

    Returns:
        Snapshot extension object.

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot`
    """

    return snapshot(target=target, filename=filename, savefun=savefun,
                    **kwargs)


def snapshot(savefun=None,
             filename='snapshot_iter_{.updater.iteration}', **kwargs):
    """snapshot(savefun=None, filename='snapshot_iter_{.updater.iteration}', \
*, target=None, condition=None, writer=None, snapshot_on_error=False, \
num_retain=-1, autoload=False)

    Returns a trainer extension to take snapshots of the trainer.

    This extension serializes the trainer object and saves it to the output
    directory. It is used to support resuming the training loop from the saved
    state.

    This extension is called once per epoch by default. To take a
    snapshot at a different interval, a trigger object specifying the
    required interval can be passed along with this extension
    to the `extend()` method of the trainer.

    The default priority is -100, which is lower than that of most
    built-in extensions.

    .. note::
       This extension first writes the serialized object to a temporary file
       and then rename it to the target file name. Thus, if the program stops
       right before the renaming, the temporary file might be left in the
       output directory.

    Args:
        savefun: Function to save the trainer. It takes two arguments: the
            output file path and the trainer object.
            It is :meth:`chainer.serializers.save_npz` by default.
            If ``writer`` is specified, this argument must be ``None``.
        filename (str): Name of the file into which the trainer is serialized.
            It can be a format string, where the trainer object is passed to
            the :meth:`str.format` method.
        target: Object to serialize. If it is not specified, it will
            be the trainer object.
        condition: Condition object. It must be a callable object that returns
            boolean without any arguments. If it returns ``True``, the snapshot
            will be done.
            If not, it will be skipped. The default is a function that always
            returns ``True``.
        writer: Writer object.
            It must be a callable object.
            See below for the list of built-in writers.
            If ``savefun`` is other than ``None``, this argument must be
            ``None``. In that case, a
            :class:`~chainer.training.extensions.snapshot_writers.SimpleWriter`
            object instantiated with specified ``savefun`` argument will be
            used.
        snapshot_on_error (bool): Whether to take a snapshot in case trainer
            loop has been failed.
        num_retain (int): Number of snapshot files to retain
            through the cleanup. Must be a positive integer for any cleanup to
            take place. Automatic deletion of old snapshots only works when the
            filename is string.
        autoload (bool): With this enabled, the extension
            automatically finds the latest snapshot and loads the data
            to the target.  Automatic loading only works when the
            filename is a string. It is assumed that snapshots are generated
            by :func:`chainer.serializers.save_npz` .

    Returns:
        Snapshot extension object.

    .. testcode::
       :hide:

       from chainer import training
       class Model(chainer.Link):
           def __call__(self, x):
               return x
       train_iter = chainer.iterators.SerialIterator([], 1)
       optimizer = optimizers.SGD().setup(Model())
       updater = training.updaters.StandardUpdater(
           train_iter, optimizer, device=0)
       trainer = training.Trainer(updater)

    .. admonition:: Using asynchronous writers

        By specifying ``writer`` argument, writing operations can be made
        asynchronous, hiding I/O overhead of snapshots.

        >>> from chainer.training import extensions
        >>> writer = extensions.snapshot_writers.ProcessWriter()
        >>> trainer.extend(extensions.snapshot(writer=writer), \
trigger=(1, 'epoch'))

        To change the format, such as npz or hdf5, you can pass a saving
        function as ``savefun`` argument of the writer.

        >>> from chainer.training import extensions
        >>> from chainer import serializers
        >>> writer = extensions.snapshot_writers.ProcessWriter(
        ...     savefun=serializers.save_npz)
        >>> trainer.extend(extensions.snapshot(writer=writer), \
trigger=(1, 'epoch'))

    This is the list of built-in snapshot writers.

        - :class:`chainer.training.extensions.snapshot_writers.SimpleWriter`
        - :class:`chainer.training.extensions.snapshot_writers.ThreadWriter`
        - :class:`chainer.training.extensions.snapshot_writers.ProcessWriter`
        - :class:`chainer.training.extensions.snapshot_writers.\
ThreadQueueWriter`
        - :class:`chainer.training.extensions.snapshot_writers.\
ProcessQueueWriter`

    .. seealso::

        - :meth:`chainer.training.extensions.snapshot_object`
    """
    target, condition, writer, snapshot_on_error, num_retain,\
        autoload = argument.parse_kwargs(
            kwargs,
            ('target', None), ('condition', None), ('writer', None),
            ('snapshot_on_error', False), ('num_retain', -1),
            ('autoload', False))
    argument.assert_kwargs_empty(kwargs)

    if savefun is not None and writer is not None:
        raise TypeError(
            'savefun and writer arguments cannot be specified together.')

    if writer is None:
        if savefun is None:
            savefun = torch.save
        writer = snapshot_writers.SimpleWriter(savefun=savefun)

    return _Snapshot(
        target=target, condition=condition, writer=writer, filename=filename,
        snapshot_on_error=snapshot_on_error, num_retain=num_retain,
        autoload=autoload)


def _always_true():
    return True


class _Snapshot(extension.Extension):
    """Trainer extension to take snapshots.

    This extension serializes the given object and saves it to the output
    directory.

    This extension is called once per epoch by default. To take a
    snapshot at a different interval, a trigger object specifying the
    required interval can be passed along with this extension
    to the `extend()` method of the trainer.

    The default priority is -100, which is lower than that of most
    built-in extensions.
    """
    trigger = 1, 'epoch'
    priority = -100

    def __init__(
            self, target=None, condition=None, writer=None,
            filename='snapshot_iter_{.updater.iteration}',
            snapshot_on_error=False, num_retain=-1, autoload=False):
        if condition is None:
            condition = _always_true
        if writer is None:
            writer = snapshot_writers.SimpleWriter()
        self._target = target
        self.filename = filename
        self.condition = condition
        self.writer = writer
        self._snapshot_on_error = snapshot_on_error
        self.num_retain = num_retain
        self.autoload = autoload

    def initialize(self, trainer):
        target = trainer if self._target is None else self._target
        outdir = trainer.out
        if self.autoload:
            # If ``autoload`` is on, this code scans the ``outdir``
            # for potential snapshot files by matching the file names
            # from ``filename`` format, picks up the latest one in
            # terms of mtime, and tries to load it it the target or
            # trainer.
            filename = _find_latest_snapshot(self.filename, outdir)
            if filename is None:
                print('No snapshot file that matches {} was found'
                      .format(self.filename))
            else:
                snapshot_file = os.path.join(outdir, filename)
                # As described above (at ``autoload`` option),
                # snapshot files to be autoloaded must be saved by
                # ``save_npz`` . In order to support general format,
                # we nned to first reconstruct the design of savefun
                # and loadfun.
                state = torch.load(snapshot_file)
                target.load_state_dict(state)

        if (hasattr(self.writer, '_add_cleanup_hook')
                and self.num_retain > 0
                and isinstance(self.filename, str)):
            # This block sets a method to automatic cleanup of stale
            # snapshots, when ``num_retain`` argument is positive
            # number. When the given snapshot writer is Chainer's
            # built-in writer, a cleanup method that is to be
            # triggered right after creation of new snapshot file, is
            # injected here.
            def _cleanup():
                files = _find_stale_snapshots(self.filename, outdir,
                                              self.num_retain)
                for file in files:
                    os.remove(os.path.join(outdir, file))

            self.writer._add_cleanup_hook(_cleanup)

    def on_error(self, trainer, exc, tb):
        super(_Snapshot, self).on_error(trainer, exc, tb)
        if self._snapshot_on_error:
            self._make_snapshot(trainer)

    def __call__(self, trainer):
        if self.condition():
            self._make_snapshot(trainer)

    def _make_snapshot(self, trainer):
        target = trainer if self._target is None else self._target
        # We need to get a dictionary with the sate here
        serialized_target = target.state_dict()
        filename = self.filename
        if callable(filename):
            filename = filename(trainer)
        else:
            filename = filename.format(trainer)
        outdir = trainer.out
        self.writer(filename, outdir, serialized_target)

    def finalize(self):
        if hasattr(self.writer, 'finalize'):
            self.writer.finalize()