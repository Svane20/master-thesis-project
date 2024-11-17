import torch

from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from constants.directories import OUTPUT_DIRECTORY
from metrics.DICE import calculate_DICE, calculate_DICE_edge


def save_predictions(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        device: torch.device,
        num_batches: int = None,
        directory: Path = OUTPUT_DIRECTORY,
) -> None:
    """
    Save the model predictions as images.

    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader
        device (torch.device): Device to use for evaluation
        num_batches (int, optional): Number of batches to process. If None, process all batches. Default is None.
        directory (Path): Directory to save the images to. Default is "output".
    """
    # Create the directory if it does not exist
    directory.mkdir(parents=True, exist_ok=True)

    model.eval()

    data_iter = iter(data_loader)
    batches_to_process = []
    for _ in range(num_batches):
        try:
            batch = next(data_iter)
            batches_to_process.append(batch)
        except StopIteration:
            break
    print(f"Processing the first {len(batches_to_process)} batch(es).")

    with torch.inference_mode():
        for batch_idx, (X, y) in tqdm(enumerate(batches_to_process), desc="Saving predictions"):
            X, y = X.to(device), y.to(device)

            y_logits = model(X)
            y_preds = torch.sigmoid(y_logits)
            preds = (y_preds > 0.5).float()

            # For each sample in the batch
            for idx in range(X.size(0)):
                input_img = X[idx].cpu().numpy().transpose(1, 2, 0)
                target_mask = y[idx].cpu().numpy().squeeze()
                pred_mask = preds[idx].cpu().numpy().squeeze()

                # Un-normalize the image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                input_img = std * input_img + mean
                input_img = np.clip(input_img, 0, 1)

                # Compute metrics per image
                dice_score = calculate_DICE(torch.tensor(pred_mask), torch.tensor(target_mask))
                dice_edge_score = calculate_DICE_edge(torch.tensor(pred_mask), torch.tensor(target_mask))

                # Create figure
                fig, ax = plt.subplots(1, 3, figsize=(18, 6))

                # Display input image
                ax[0].imshow(input_img)
                ax[0].set_title('Input Image')
                ax[0].axis('off')

                # Display ground truth mask
                ax[1].imshow(target_mask, cmap='gray')
                ax[1].set_title('Ground Truth Mask')
                ax[1].axis('off')

                # Display prediction overlay
                ax[2].imshow(input_img)
                ax[2].imshow(pred_mask, cmap='jet', alpha=0.5)
                ax[2].set_title(f'Predicted Mask Overlay\nDice: {dice_score:.4f}, Dice Edge: {dice_edge_score:.4f}')
                ax[2].axis('off')

                sample_idx = batch_idx * data_loader.batch_size + idx
                sample_path = directory / f"sample_{sample_idx}.png"

                plt.tight_layout()
                plt.subplots_adjust(top=0.85)
                plt.savefig(sample_path)
                plt.close(fig)
