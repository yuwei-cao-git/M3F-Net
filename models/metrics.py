import torch

def r2_score_torch(y_true, y_pred):
    """
    Compute the R² score in PyTorch to avoid moving tensors to CPU.
    
    Args:
    - y_true: Ground truth tensor (valid pixels, num_channels).
    - y_pred: Predicted tensor (valid pixels, num_channels).

    Returns:
    - r2: The R² score computed in PyTorch.
    """
    # Mean of the true values
    y_true_mean = torch.mean(y_true, dim=0)

    # Total sum of squares (TSS)
    total_variance = torch.sum((y_true - y_true_mean) ** 2, dim=0)

    # Residual sum of squares (RSS)
    residuals = torch.sum((y_true - y_pred) ** 2, dim=0)

    # To handle the case where total_variance is zero (i.e., constant target values),
    # we use torch.where to define R² as 0 in these cases.
    r2 = torch.where(total_variance != 0, 1 - (residuals / total_variance), torch.tensor(0.0, device=y_true.device))

    return r2.mean()  # Mean R² across all channels

def f1_score_torch(y_true, y_pred, num_classes):
    """
    Computes the F1 score for multiclass classification.

    Args:
        y_true: Ground truth class labels (num_samples,).
        y_pred: Predicted class labels (num_samples,).
        num_classes: Total number of classes.

    Returns:
        f1_score: The average F1 score across all classes.
    """
    f1_scores = []
    for cls in range(num_classes):
        true_positive = ((y_pred == cls) & (y_true == cls)).sum().float()
        false_positive = ((y_pred == cls) & (y_true != cls)).sum().float()
        false_negative = ((y_pred != cls) & (y_true == cls)).sum().float()

        precision = true_positive / (true_positive + false_positive + 1e-6)
        recall = true_positive / (true_positive + false_negative + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        f1_scores.append(f1)

    # Average F1 score over classes
    f1_score = torch.stack(f1_scores).mean()
    return f1_score