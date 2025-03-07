import torch
import torch.nn as nn

class DebiasedBatchNorm(nn.Module):
    """
    DebiasedBatchNorm layer:
      - Normalizes inputs using statistics computed from a selected sub-batch.
      - Assumes input x of shape (B, C) or (B, C, L).
      - Requires that the batch size B is exactly equal to sub_batch_num * sub_batch_size.

    Args:
        num_channels (int): Number of channels (C).
        sub_batch_num (int): Number of sub-batches to split the batch into. Must be at least 3.
        sub_batch_size (int): Size of each sub-batch.
        eps (float): Small constant for numerical stability.
        momentum (float): Momentum for running stats update.
        affine (bool): If True, uses learnable affine parameters.
        track_running_stats (bool): If True, tracks running mean and variance.
    """
    def __init__(self, num_channels, sub_batch_num, sub_batch_size, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(DebiasedBatchNorm, self).__init__()
        if sub_batch_num < 3:
            raise ValueError("sub_batch_num must be at least 3 for meaningful outlier selection.")
        self.num_channels = num_channels
        self.sub_batch_num = sub_batch_num
        self.sub_batch_size = sub_batch_size
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_channels))
            self.register_buffer("running_var", torch.ones(num_channels))

    def compute_stats(self, x, dims, unbiased: bool):
        mean = x.mean(dim=dims)
        var = x.var(dim=dims, unbiased=unbiased)
        return mean, var

    def reshape_input(self, x: torch.Tensor):
        """
        Reshape input to ensure it has shape (B, C, L). If input is (B, C), reshapes to (B, C, 1).
        """
        original_shape = x.shape
        if len(original_shape) == 2:
            x = x.unsqueeze(-1)
        elif len(original_shape) == 3:
            pass
        else:
            raise ValueError("Input must have shape (B, C) or (B, C, L).")
        return x, original_shape

    def forward(self, x: torch.Tensor):
        # Reshape input so that it always has shape (B, C, L)
        x, original_shape = self.reshape_input(x)
        B, C, L = x.shape
        expected_batch_size = self.sub_batch_num * self.sub_batch_size
        if B != expected_batch_size:
            raise ValueError(f"Input batch size ({B}) must equal sub_batch_num * sub_batch_size ({expected_batch_size}).")
        
        if self.training:
            # Reshape x into (sub_batch_num, sub_batch_size, C, L)
            x_4d = x.view(self.sub_batch_num, self.sub_batch_size, C, L)
            # Compute statistics for each sub-batch along the sub-batch dimension and sequence length (dim 1 and 3)
            sub_means, sub_vars = self.compute_stats(x_4d, dims=(1, 3), unbiased=False)  # shape: (sub_batch_num, C)
            # Compute pairwise distances between sub-batch means
            diff_means = sub_means.unsqueeze(1) - sub_means.unsqueeze(0)  # (sub_batch_num, sub_batch_num, C)
            dist_sq = diff_means.pow(2).mean(dim=-1).sqrt()  # (sub_batch_num, sub_batch_num)
            am_dist = dist_sq.mean(dim=-1)  # (sub_batch_num)
            chosen_idx = torch.argmax(am_dist)
            
            # Use the chosen sub-batch statistics to normalize the entire batch
            mean, var = sub_means[chosen_idx], sub_vars[chosen_idx]
            x_norm = (x - mean.view(1, C, 1)) / torch.sqrt(var.view(1, C, 1) + self.eps)
            
            # Update running statistics
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean.copy_((1 - self.momentum) * self.running_mean + self.momentum * mean)
                    self.running_var.copy_((1 - self.momentum) * self.running_var + self.momentum * var)
        else:
            # Use running statistics during evaluation
            rm = self.running_mean.view(1, C, 1)
            rv = self.running_var.view(1, C, 1)
            x_norm = (x - rm) / torch.sqrt(rv + self.eps)
        
        if self.affine:
            w = self.weight.view(1, C, 1)
            b = self.bias.view(1, C, 1)
            x_norm = x_norm * w + b

        if len(original_shape) == 2:
            x_norm = x_norm.squeeze(-1)
        return x_norm
