import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import shutil
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bptt = 35

def batchify(data, bsz: int):
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

def get_batch(source, i: int):
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def get_random_batch_indices(data_size, seq_len,seed = None):
    """
    Generate random starting indices for batch selection.

    Args:
        data_size: int, the total size of the dataset.
        seq_len: int, the desired sequence length for each batch.
        bptt: int, backpropagation through time window size.
        batch_size: int, the number of sequences per batch.

    Returns:
        A list of random starting indices.
    """
    if seed is not None:
        torch.manual_seed(seed)  # Set the seed for reproducibility

    indices = []
    i = 0
    while i < data_size - 1:
        seq_len = min(bptt, data_size - 1 - i)
        indices.append(i)
        i += seq_len  # Move i by the seq_len just calculated

    # Ensure randomness in the selection of starting indices
    # max_start_index = data_size - seq_len - 1
    num_batches = len(indices)
    if num_batches > 0:
        # Generate a random permutation of indices and select the first `num_batches` indices
        perm = torch.randperm(len(indices))
        selected_indices = perm[:num_batches]
        random_indices = [indices[i] for i in selected_indices]
    else:
        random_indices = []

    return random_indices

def save_checkpoint(model, epoch, optimizer = None):
    model_dir = f"checkpoint/model_{model.id}"
    if epoch ==1:
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            # List all files and directories in the directory
            for filename in os.listdir(model_dir):
                file_path = os.path.join(model_dir, filename)
                try:
                    # If it's a file, remove it
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    # If it's a directory, remove it and all its contents
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')
    # Check if the parent directory exists, if not, create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model state dict
    model_path = os.path.join(model_dir, f"checkpoint_{model.id}_{epoch}.pt")
    torch.save({"model_state": model.state_dict(), "optimizer":  optimizer}, model_path)

    