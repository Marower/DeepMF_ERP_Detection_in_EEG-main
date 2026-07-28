import torch
import torch.nn as nn
import torch.nn.functional as F

class TraditionalMatchedFilter(nn.Module):
    def __init__(self, templates, normalize=True):
        """
        templates: PyTorch Tensor or NumPy array of shape (4, template_length)
                   or (1, 4, template_length).
        normalize: If True, divides template by its L2 norm to get unit energy.
        """
        super().__init__()
        
        if not isinstance(templates, torch.Tensor):
            templates = torch.tensor(templates, dtype=torch.float32)
            
        # Ensure shape is (out_channels, in_channels/groups, kernel_length) -> (4, 1, template_length)
        if templates.ndim == 2:  # Shape: (4, template_length)
            templates = templates.unsqueeze(1)
        elif templates.ndim == 3 and templates.shape[0] == 1:  # Shape: (1, 4, template_length)
            templates = templates.squeeze(0).unsqueeze(1)

        # Normalize templates to equalize channel energy
        if normalize:
            norm = torch.norm(templates, dim=-1, keepdim=True) + 1e-8
            templates = templates / norm

        # Time-reverse templates along time dimension for cross-correlation
        kernel = torch.flip(templates, dims=[-1])
        
        # Register as fixed buffer (no gradients computed)
        self.register_buffer('kernel', kernel)

    def forward(self, x):
        """
        x shape: (batch_size, 4, time_samples) -> e.g., (1, 4, 500)
        returns shape: (batch_size, time_samples) -> aggregated output across channels
        """
        # Apply 1D Depthwise Convolution (groups=4 applies channel 0 template to channel 0 input, etc.)
        filtered = F.conv1d(x, self.kernel, padding='same', groups=4)
        
        # Aggregate filtered responses across the 4 channels (sum or mean)
        # to match single 1D response expected by find_peaks
        output_MF, _ = torch.max(filtered, dim=1)
        
        return output_MF