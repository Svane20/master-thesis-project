import torch
from torchmetrics import ConfusionMatrix

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import mlxtend.plotting as mlxplt
from typing import Literal, List


def plot_confusion_matrix(
        class_names: List[str],
        y_preds: torch.Tensor,
        targets: torch.Tensor,
        task: Literal["binary", "multiclass", "multilabel"] = "multiclass"
) -> None:
    """
    Plot the confusion matrix.

    Args:
        class_names (List[str]): List of class names.
        y_preds (torch.Tensor): Predictions.
        targets (torch.Tensor): Targets.
        task (Literal["binary", "multiclass", "multilabel"], optional): Task type. Defaults to "multiclass".
    """
    confmat = ConfusionMatrix(num_classes=len(class_names), task=task)
    confmat_tensor = confmat(y_preds, targets)

    fig, ax = mlxplt.plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),
        figsize=(10, 10),
        class_names=class_names,
    )
    plt.show()


def plot_prediction(
        class_names: List[str],
        image: Image.Image,
        label: int,
        probabilities: np.ndarray,
):
    """
    Plot the prediction.

    Args:
        class_names (List[str]): List of class names.
        image (Image.Image): Image.
        label (int): The predicted label.
        probabilities (np.ndarray): The predictions of the image.
    """
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title(f"Prediction: {class_names[label]} | Probability: {probabilities.max():.3f}")
    plt.axis("off")
    plt.show()
