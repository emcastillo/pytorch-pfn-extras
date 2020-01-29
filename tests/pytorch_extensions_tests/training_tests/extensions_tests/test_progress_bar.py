import io

import pytorch_extensions as pe


def test_run():
    max_epochs = 10
    epoch_size = 10
    manager = pe.training.ExtensionsManager({}, [], max_epochs, [])

    out = io.StringIO()
    extension = pe.training.extensions.ProgressBar(
        training_length=None,
        update_interval=1,
        bar_length=40,
        out=out,
    )
    manager.extend(extension)

    for epoch in range(max_epochs):
        for batch_idx in range(epoch_size):
            cur_it = epoch * epoch_size + batch_idx
            with manager.run_iteration(
                    iteration=cur_it, epoch_size=epoch_size):
                if cur_it < 2:
                    continue
                status = '{} iter, {} epoch / {} epochs'.format(
                    cur_it, epoch, max_epochs)
                assert status in out.getvalue()
