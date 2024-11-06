import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedMSELoss(nn.Module):
    def __init__(self, weights):
        super(WeightedMSELoss, self).__init__()
        self.weights = weights

    def forward(self, y_pred, y_true):
        squared_errors = torch.square(y_pred - y_true)
        weighted_squared_errors = squared_errors * self.weights
        loss = torch.mean(
            weighted_squared_errors
        )  # for multi-gpu, should it set to sum?
        return loss


def calc_loss(y_true, y_pred, weights):
    weighted_mse = WeightedMSELoss(weights)
    loss = weighted_mse(y_pred, y_true)

    return loss


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, outputs, targets, mask):
        """
        Custom MSE loss function that ignores NoData pixels.

        Args:
        - outputs: Predicted values (batch_size, num_channels, H, W)
        - targets: Ground truth values (batch_size, num_channels, H, W)
        - mask: Boolean mask indicating NoData pixels (batch_size, H, W)

        Returns:
        - loss: Mean squared error computed only for valid pixels.
        """
        # Expand mask to match the shape of outputs and targets
        expanded_mask = mask.unsqueeze(1).expand_as(
            outputs
        )  # Shape: (batch_size, num_channels, H, W)

        # Compute squared differences, applying mask to ignore invalid pixels
        diff = (outputs - targets) ** 2
        valid_diff = diff * (
            ~expanded_mask
        )  # Keep only valid pixels (where mask is False)

        # Sum over the channel and spatial dimensions (H, W)
        loss = valid_diff.sum(dim=(1, 2, 3))

        # Count the number of valid pixels per batch (sum of ~mask)
        num_valid_pixels = (~expanded_mask).sum(dim=(1, 2, 3)).float()

        # Prevent division by zero (if all pixels are NoData)
        num_valid_pixels = torch.clamp(num_valid_pixels, min=1.0)

        # Compute mean squared error per valid pixel
        loss = loss / num_valid_pixels

        # Return the average loss over the batch
        return loss.mean()


# create a nn class (just-for-fun choice :-)
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = MaskedMSELoss()

    def forward(self, outputs, targets, mask):
        return torch.sqrt(self.mse(outputs, targets, mask))


def apply_mask(outputs, targets, mask, multi_class=True):
    """
    Applies the mask to outputs and targets to exclude invalid data points.

    Args:
        outputs: Model predictions (batch_size, num_classes, H, W) for images or (batch_size, num_points, num_classes) for point clouds.
        targets: Ground truth labels (same shape as outputs).
        mask: Boolean mask indicating invalid data points (True for invalid).

    Returns:
        valid_outputs: Masked and reshaped outputs.
        valid_targets: Masked and reshaped targets.
    """
    # Expand the mask to match outputs and targets
    if multi_class:
        expanded_mask = mask.unsqueeze(1).expand_as(
            outputs
        )  # Shape: (batch_size, num_classes, H, W)
        num_classes = outputs.size(1)
    else:
        expanded_mask = mask

    # Apply mask to exclude invalid data points
    valid_outputs = outputs[~expanded_mask]
    valid_targets = targets[~expanded_mask]
    # Reshape to (-1, num_classes)
    if multi_class:
        valid_outputs = valid_outputs.view(-1, num_classes)
        valid_targets = valid_targets.view(-1, num_classes)

    return valid_outputs, valid_targets
