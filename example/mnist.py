from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import pytorch_extensions as pte
import pytorch_extensions.training.extensions as extensions

# Extensions manager object
manager = None


class Net(nn.Module):
    def __init__(self, lazy):
        super(Net, self).__init__()
        if lazy:
            self.conv1 = pte.nn.LazyConv2d(None, 20, 5, 1)
            self.conv2 = pte.nn.LazyConv2d(None, 50, 5, 1)
            self.fc1 = pte.nn.LazyLinear(None, 500)
            self.fc2 = pte.nn.LazyLinear(None, 10)
        else:
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4*4*50, 500)
            self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    epoch_size = len(train_loader)
    for batch_idx, (data, target) in enumerate(train_loader):
        current_it = epoch * epoch_size + batch_idx
        with manager.run_iteration(
                iteration=current_it, epoch_size=epoch_size):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            pte.reporter.report({'train/loss': loss.item()})
            loss.backward()
            optimizer.step()


def test(args, model, device, data, target):
    """ The extension loops over the iterator in order to
        drive the evaluator progress bar and reporting
        averages
    """
    model.eval()
    test_loss = 0
    correct = 0
    data, target = data.cuda(), target.cuda()
    output = model(data)
    # Final result will be average of averages of the same size
    test_loss += F.nll_loss(output, target, reduction='mean').item()
    pte.reporter.report({'val/loss': test_loss})
    pred = output.argmax(dim=1, keepdim=True)
    correct += pred.eq(target.view_as(pred)).sum().item()
    pte.reporter.report({'val/acc': correct/len(data)})


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000,
                        metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', dest='cuda',
                        action='store_false', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--snapshot', type=str, default=None,
                        help='path to snapshot file')
    parser.add_argument('--no-lazy', dest='lazy',
                        action='store_false', default=True,
                        help='do not use lazy modules')
    args = parser.parse_args()
    use_cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net(args.lazy)
    if args.lazy:
        # You need to run a dummy forward to initialize parameters.
        # This should be done before passing parameter list to optimizers.
        dummy_input = train_loader.dataset[0][0].unsqueeze(0)
        model(dummy_input)
    model.to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum)

    global manager
    # manager.extend(...) also works
    writer = extensions.snapshot_writers.SimpleWriter()
    my_extensions = [
        extensions.LogReport(),
        extensions.ProgressBar(),
        extensions.ExponentialShift(
            'lr', 0.9999, optimizer, init=0.2, target=0.1),
        extensions.observe_lr(optimizer=optimizer),
        extensions.ParameterStatistics(model, prefix='model'),
        extensions.VariableStatisticsPlot(model),
        extensions.Evaluator(
            test_loader, model,
            eval_func=lambda data, target:
                test(args, model, device, data, target),
            progress_bar=True),
        extensions.PlotReport(
            ['train/loss', 'val/loss'], 'epoch', filename='loss.png'),
        extensions.PrintReport(['epoch', 'iteration',
                                'train/loss', 'lr', 'model/fc2.bias/grad/min',
                                'val/loss', 'val/acc']),
        extensions.snapshot(writer=writer),
    ]
    models = {'main': model}
    optimizers = {'main': optimizer}
    manager = pte.training.ExtensionsManager(
        models, optimizers, args.epochs, my_extensions)
    # Lets load the snapshot
    if args.snapshot is not None:
        state = torch.load(args.snapshot)
        manager.load_state_dict(state)
    for epoch in range(0, args.epochs):
        train(args, model, device, train_loader, optimizer, epoch)
        # Test function is called from the evaluator extension
        # to get access to the reporter and other facilities
        # test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
