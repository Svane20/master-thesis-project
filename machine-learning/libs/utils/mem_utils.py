import torch

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