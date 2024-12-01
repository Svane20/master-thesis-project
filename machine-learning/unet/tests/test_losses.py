import torch

from training.custom_criterions import EdgeWeightedBCEDiceLoss


def test_edge_weighted_bce_dice_loss():
    loss_fn = EdgeWeightedBCEDiceLoss()
    inputs = torch.randn((1, 1, 10, 10), requires_grad=True)
    targets = torch.randint(0, 2, (1, 1, 10, 10)).float()
    loss = loss_fn(inputs, targets)
    assert loss.item() >= 0, "Loss should be non-negative"
