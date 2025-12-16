import numpy as np
import re
import os

def find_frames(search_dir, pattern,min_frame=0,max_frame=-1):
    """
    Find files matching a regex pattern with a captured numeric frame index.

    Args:
        search_dir (str): Directory to search
        pattern (str): Regex pattern with ONE capture group for frame number

    Returns:
        frames (list[str]): Sorted list of matching file paths
        frame_range (tuple[int, int] | None): (min_frame, max_frame) or None
    """

    if not (isinstance(search_dir, str) and os.path.isdir(search_dir)):
        raise NotImplementedError("Search path is not a string and directory.")   
    else:
        min_frame = int(min_frame)
        max_frame = int(max_frame)
        print(f"\nSearching \'{search_dir}\' for frames using regex pattern \'{pattern}\'...")
        if min_frame <= max_frame:
            print(f"Restricting Frame Range : [{min_frame}-{max_frame}]")
        else:
            print(f"Gathering all frame numbers >= {min_frame}")

        regex = re.compile(pattern)

        matches = []
        frame_numbers = []

        for fname in os.listdir(search_dir):
            match = regex.match(fname)
            if not match:
                continue

            frame_num = int(match.group(1))
            if frame_num >= min_frame: # Keep if >= min frame 
                if max_frame >= min_frame: # Should enforce max frame?
                    if frame_num <= max_frame: # Keep if <= max frame
                        matches.append(os.path.join(search_dir, fname))
                        frame_numbers.append(frame_num)
                else: # No max frame, keep all >= min frame
                    matches.append(os.path.join(search_dir, fname))
                    frame_numbers.append(frame_num)

        if not frame_numbers:
            print("ERROR: No frames found.")
            return [], None

        # sort by frame number
        matches = [x for _, x in sorted(zip(frame_numbers, matches))]
        frame_numbers.sort()

        # print(f"MATCHES: {matches}")
        # print(f"FRAME_NUMBERS: {frame_numbers}")
        print(f"Found {len(matches)} frames with range [{frame_numbers[0]}-{frame_numbers[-1]}]\n")
        return matches, (frame_numbers[0], frame_numbers[-1])

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

# RUN TESTS
if __name__ == "__main__":
    search_folder = "/home/csrobot/Desktop/ATB1_BENCHMARK/mi6ni"
    file_pattern = r"^frame_(\d{5})\.jpg$"
    frames, range = find_frames(search_folder,file_pattern)

    print(frames)
    print(range)