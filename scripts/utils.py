import numpy as np

def mask_to_bbox(mask: np.ndarray):
    """
    Given a 2D boolean mask, return (x_min, y_min, x_max, y_max) in pixel coords.
    Returns None if mask is empty.
    """
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        raise ValueError("mask_to_bbox expects a 2D boolean array")

    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None  # no mask content

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return (x_min, y_min, x_max, y_max)

def compute_iou(box1, box2):
    """Compute IoU between two YOLO-format boxes (xc, yc, w, h)."""
    # Convert to corner coordinates
    def to_xy(box):
        xc, yc, w, h = box
        return [
            xc - w / 2,
            yc - h / 2,
            xc + w / 2,
            yc + h / 2,
        ]

    x1_min, y1_min, x1_max, y1_max = to_xy(box1)
    x2_min, y2_min, x2_max, y2_max = to_xy(box2)

    # Intersection box
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union = area1 + area2 - inter_area

    return inter_area / union if union > 0 else 0.0