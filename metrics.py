def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Compute the Dice Coefficient for evaluating segmentation performance.
    """
    y_true_flat = y_true.view(-1)
    y_pred_flat = y_pred.view(-1)
    intersection = (y_true_flat * y_pred_flat).sum()
    return (2.0 * intersection + smooth) / (y_true_flat.sum() + y_pred_flat.sum() + smooth)

def iou(y_true, y_pred, smooth=1e-6):
    """
    Compute Intersection over Union (IoU) for evaluating segmentation performance.
    """
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + smooth) / (union + smooth)
