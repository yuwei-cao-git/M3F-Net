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

    def forward(self, output, target, nodata_mask):
        """
        Computes per-pixel loss, masking out the nodata regions.
        """
        # output: Shape (batch_size, num_classes, height, width)
        # target: Shape (batch_size, num_classes, height, width)
        # nodata_mask: Shape (batch_size, height, width)

        # Apply the nodata mask to ignore invalid pixels
        valid_mask = ~nodata_mask.unsqueeze(1)  # Shape: (batch_size, 1, height, width)
        valid_mask = valid_mask.expand_as(
            output
        )  # Shape: (batch_size, num_classes, height, width)

        # Compute loss per pixel
        loss = F.mse_loss(
            output * valid_mask.float(), target * valid_mask.float(), reduction="sum"
        )

        # Compute the average loss over valid pixels
        num_valid_pixels = valid_mask.float().sum()
        if num_valid_pixels > 0:
            loss = loss / num_valid_pixels
        else:
            loss = torch.tensor(0.0, requires_grad=True).to(output.device)

        return loss


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
