import pytest
import torch

from pytorch_pfn_extras import training


@pytest.mark.gpu
def test_transfer_data():
    batch = [torch.randn(2, 2), torch.randn(3, 4)]
    assert all(batch[i].device == torch.device('cpu') for i in range(2))
    assert all(batch[i].device != torch.device('cuda:0') for i in range(2))

    batch0 = training.transfer_data(batch, 'cuda:0')
    assert all(batch0[i].device == torch.device('cuda:0') for i in range(2))
    assert all(batch0[i].device != torch.device('cpu') for i in range(2))

    batch1 = training.transfer_data(batch, device='cuda:0')
    assert all(batch1[i].device == torch.device('cuda:0') for i in range(2))
    assert all(batch1[i].device != torch.device('cpu') for i in range(2))
