import torch

from metrics.DICE import calculate_DICE, calculate_DICE_edge
from utils.edge_detection import compute_edge_map


def test_calculate_DICE_perfect_match():
    preds = torch.ones((1, 1, 10, 10))
    targets = torch.ones((1, 1, 10, 10))
    dice_score = calculate_DICE(preds, targets)
    assert dice_score == 1.0, f"Expected DICE score of 1.0, got {dice_score}"


def test_calculate_DICE_no_overlap():
    preds = torch.zeros((1, 1, 10, 10))
    targets = torch.ones((1, 1, 10, 10))
    dice_score = calculate_DICE(preds, targets)
    assert abs(dice_score - 0.0) < 1e-6, f"Expected DICE score close to 0.0, got {dice_score}"


def test_calculate_DICE_edge_perfect_match():
    preds = torch.ones((1, 1, 10, 10))
    targets = torch.ones((1, 1, 10, 10))
    dice_edge_score = calculate_DICE_edge(preds, targets)
    assert dice_edge_score == 1.0, f"Expected DICE edge score of 1.0, got {dice_edge_score}"


def test_compute_edge_map():
    tensor = torch.zeros((1, 1, 10, 10))
    tensor[:, :, 5, :] = 1  # Add a horizontal line
    edge_map = compute_edge_map(tensor)
    assert edge_map.max() > 0, "Edge map should detect edges"


def test_calculate_DICE_edge_2D_image():
    preds = (torch.randn(100, 100) > 0.5).float()
    targets = (torch.randn(100, 100) > 0.5).float()
    dice_edge_score = calculate_DICE_edge(preds, targets)
    assert 0.0 <= dice_edge_score <= 1.0, f"Dice score for single 2D image is out of range: {dice_edge_score}"


def test_calculate_DICE_edge_3D_tensor():
    preds = (torch.randn(1, 100, 100) > 0.5).float()  # Single RGB image
    targets = (torch.randn(1, 100, 100) > 0.5).float()  # Single RGB ground truth
    dice_edge_score = calculate_DICE_edge(preds, targets)
    assert 0.0 <= dice_edge_score <= 1.0, f"Dice score for single 3D tensor is out of range: {dice_edge_score}"


def test_calculate_DICE_edge_4D_batch():
    preds = (torch.randn(2, 1, 100, 100) > 0.5).float()  # Batch of 2 predictions
    targets = (torch.randn(2, 1, 100, 100) > 0.5).float()  # Batch of 2 ground truth masks
    dice_edge_score = calculate_DICE_edge(preds, targets)
    assert 0.0 <= dice_edge_score <= 1.0, f"Dice score for batch of 4D tensors is out of range: {dice_edge_score}"
