import torch
import cProfile
import pstats
import io
import numpy as np
import math


def estimate_max_batch_size(
        model,
        input_size=(3, 512, 512),
        max_memory_gb=10.0,
        safety_factor=0.9,
        input_generator_fn=None,
):
    """
    Estimate the maximum batch size that fits in the given GPU memory.

    Args:
        model (torch.nn.Module): The model to evaluate.
        input_size (tuple): The input size (C, H, W) for both inputs.
        max_memory_gb (float): Total GPU memory available (in GB).
        safety_factor (float): Factor to leave memory headroom.
        input_generator_fn (callable): Function to generate model inputs given batch size.

    Returns:
        int: Estimated max batch size.
    """
    assert input_generator_fn is not None, "You must provide an input_generator_fn for multi-input models"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    memory_per_sample = []
    for bs in [1, 2, 4]:
        torch.cuda.empty_cache()
        inputs = input_generator_fn(bs, input_size, device)
        try:
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=True, dtype=torch.float16):
                _ = model(*inputs)
            allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # in GB
            memory_per_sample.append(allocated / bs)
        except RuntimeError:
            break

    if not memory_per_sample:
        return 1

    avg_per_sample = sum(memory_per_sample) / len(memory_per_sample)
    safe_memory = max_memory_gb * safety_factor
    max_batch_size = int(safe_memory // avg_per_sample)

    return max(1, max_batch_size)


def profile_large_dataset(transforms_pipeline,
                          n_images=10000,
                          batch_size=16,
                          image_shape=(1024, 1024, 3)):
    """
    Profile the given transforms pipeline by simulating n_images of random data,
    processed in batches of batch_size. Each image has the specified image_shape.

    :param transforms_pipeline: A transform pipeline (e.g., T.Compose([...])) that
                                accepts {"image": np.array, "alpha": np.array} dicts.
    :param n_images: Total number of images to simulate (default 10,000).
    :param batch_size: Number of images per batch (default 16).
    :param image_shape: Tuple specifying the (H, W, C) shape of each image.
    """
    n_batches = math.ceil(n_images / batch_size)
    height, width, channels = image_shape

    pr = cProfile.Profile()
    pr.enable()

    for batch_idx in range(n_batches):
        # The last batch might have fewer samples if n_images isn't multiple of batch_size
        current_batch_size = min(batch_size, n_images - batch_idx * batch_size)

        # Simulate a batch of images and alpha masks
        # Shape of image_batch: (current_batch_size, H, W, C)
        image_batch = np.random.randint(
            0, 256,
            (current_batch_size, height, width, channels),
            dtype=np.uint8
        )

        # Shape of alpha_batch: (current_batch_size, H, W)
        alpha_batch = np.random.randint(
            0, 256,
            (current_batch_size, height, width),
            dtype=np.uint8
        )

        # Apply transforms to each sample individually
        for i in range(current_batch_size):
            sample = {"image": image_batch[i], "alpha": alpha_batch[i]}
            transforms_pipeline(sample)

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME)
    ps.print_stats()
    print(s.getvalue())


def profile_function(func, num_iter: int = 10):
    """
    Profile a single function (or callable) for num_iter iterations, printing stats.
    """
    pr = cProfile.Profile()
    pr.enable()

    for _ in range(num_iter):
        func()

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.TIME)
    ps.print_stats()
    print(s.getvalue())


def profile_pipeline(transforms_pipeline, sample, num_iter=10):
    """
    Profile the *entire* pipeline for num_iter iterations.
    """

    def _apply_all():
        # We copy the sample each time so transforms always see the same input
        local_sample = {
            k: (v.copy() if isinstance(v, np.ndarray) else v)
            for k, v in sample.items()
        }
        transforms_pipeline(local_sample)

    print("=== Profiling the ENTIRE transform pipeline ===")
    profile_function(_apply_all, num_iter=num_iter)


def profile_each_transform(transforms_list, sample, num_iter=10):
    """
    Profile each transform in the list individually for num_iter iterations.
    """
    for idx, transform_obj in enumerate(transforms_list):
        transform_name = transform_obj.__class__.__name__
        print(f"=== Profiling transform #{idx + 1}: {transform_name} ===")

        def _apply_one():
            local_sample = {
                k: (v.copy() if isinstance(v, np.ndarray) else v)
                for k, v in sample.items()
            }
            transform_obj(local_sample)

        profile_function(_apply_one, num_iter=num_iter)
