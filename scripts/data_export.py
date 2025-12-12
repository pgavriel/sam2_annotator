from utils import compute_iou, mask_to_bbox
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

def remove_duplicate_annotations(existing_ann, new_ann, overlap_thresh=0.5):
    """
    Removes duplicates based on label and IoU overlap threshold.
    annotations: list of (label, xc, yc, w, h)
    """
    filtered = []

    for ann in new_ann:
        keep = True
        remove_existing = []
        for e, existing in enumerate(existing_ann):
            if ann[0] == existing[0]:  # same label
                iou = compute_iou(ann[1:], existing[1:])
                dbg_str = f"⚠️   Duplicate label: {ann[0]} - IOU: {iou:.2f} - Thresh: {overlap_thresh} "
                if iou > overlap_thresh:
                    dbg_str += "[REPLACED OLD]"
                    keep = False
                else:
                    dbg_str += "[APPENDED]"
                print(dbg_str)
        if keep: # Not enough overlap for conflict, simply add to the list
            filtered.append(ann)
        else: # Still keep, but remove old overlapping label (overwrite)
            for i in remove_existing:
                del existing_ann[i]
            filtered.append(ann)

    return existing_ann + filtered

def save_yolo_annotations(
    image_path: str,
    labels: dict,
    output_dir: str,
    image_size: tuple,
    overlap_thresh: float = 0.8
):
    """
    Save or append YOLO-format bounding box annotations for a given image.

    Args:
        image_path: path to the image file (used to derive annotation filename)
        labels: dict of [class name (string)] : bbox (tuple)(x1, y1, x2, y2) pixel coordinates
        output_dir: directory to write the annotation file
        image_size: (width, height) of the image for normalization
        overlap_thresh: IoU threshold above which to remove duplicate entries
    """

    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(image_path))[0]
    anno_file = f"{img_name}.txt"
    anno_path = os.path.join(output_dir, anno_file)
    w_img, h_img = image_size

    new_annotations = []
    for lbl, bbox in labels.items():

        x1, y1, x2, y2 = bbox

        # Convert to YOLO normalized format
        x_center = ((x1 + x2) / 2) / w_img
        y_center = ((y1 + y2) / 2) / h_img
        width = abs(x2 - x1) / w_img
        height = abs(y2 - y1) / h_img

        new_annotations.append((lbl, x_center, y_center, width, height))

    # print(f"New Annotations: {new_annotations}")
    # Load existing annotations if present
    existing = []
    if os.path.exists(anno_path):
        with open(anno_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    existing.append((parts[0], *map(float, parts[1:])))

    # Append and deduplicate
    merged = remove_duplicate_annotations(existing, new_annotations, overlap_thresh)

    # Write updated file
    with open(anno_path, "w") as f:
        for lbl, xc, yc, w, h in merged:
            f.write(f"{lbl} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    # print(f"Annotation saved: {anno_path}")
    return anno_path

def export_yolo_annotations(sam2, output_dir=None, image_dir=None, overlap_thresh=0.8):
    """
    Iterate through all video segments and save YOLO-style bounding boxes.

    Args:
        sam2: Sam2_Handler object for accessing current state
        output_dir: folder to store YOLO .txt files
        image_dir: folder containing the original image files
        overlap_thresh: IoU threshold for deduplication
    """
    print("\nExporting Annotations...")
    # global video_segments, frame_names
    # image_dir = "/workspace/data"
    output_dir = output_dir +"/labels" #TODO: HARDCODED FOR TESTING, fix later
    from PIL import Image

    for frame_idx, obj_masks in sam2.video_segments.items():
        image_name = sam2.frame_names[frame_idx]
        image_path = os.path.join(image_dir, image_name)
        print(f"Frame ID: {frame_idx} -> Image path: {image_path}")

        # Load image to get size
        with Image.open(image_path) as im:
            width, height = im.size

        new_annotations = {}
        for label, mask in obj_masks.items():
            bbox = mask_to_bbox(mask)
            if bbox is None:
                continue  # skip empty masks
            new_annotations[label] = bbox
        print(f" > Labels: {list(new_annotations.keys())}")
        anno_file = save_yolo_annotations(
            image_path=image_path,
            labels=new_annotations,
            output_dir=output_dir,
            image_size=(width, height),
            overlap_thresh=overlap_thresh
        )

        print(f"✅ Exported annotations for {image_name} -> {anno_file}")

def export_binary_masks(sam2, root_folder, folder_name="masks",separate=False,unique_id=True):
    output_dir = root_folder + "/" + folder_name
    print(f"Exporting Binary Masks to: {output_dir}")
    # Establish output directory
    if os.path.isdir(output_dir):
        print(f"The directory '{output_dir}' exists.")
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"The directory '{output_dir}' has been created.")

    # For each frame with mask annotations (video_segments)
        # Combine all binary masks, export
    
    for n, img_file in enumerate(tqdm(sam2.frame_names, desc="Exporting Binary Masks")):
        masks = sam2.video_segments.get(n,{})
        if masks == {}: continue

        first_mask = next(iter(masks.values()))
        print(f"FMS:{first_mask.shape}")
        _, H, W = first_mask.shape
        combined = np.zeros((H, W), dtype=np.uint8)

        for label, mask in masks.items():
            mask = np.squeeze(mask)
            if unique_id:
                lbl_id = sam2.inference_state["obj_id_to_idx"][label]
                combined[mask] = 255-int(lbl_id)
            else:
                combined[mask] = 255
        mask_img = Image.fromarray(combined)
        mask_img.save(os.path.join(output_dir, os.path.splitext(img_file)[0] + ".png"))
        # for label, mask in masks.items():
        #     mask = mask.astype(np.uint8) * 255  # boolean → uint8
        #     mask_img = Image.fromarray(mask)
        #     if separate:
        #         mask_path = os.path.join(output_dir, f"{label}.png")
        #         mask_img.save(mask_path)
        

def export_debug_masks(sam2, root_folder, folder_name="debug_masks"):
    # global frame_names
    output_dir = root_folder + "/" + folder_name
    print(f"Exporting Colored Debug Masks to: {output_dir}")
    # Establish output directory
    if os.path.isdir(output_dir):
        print(f"The directory '{output_dir}' exists.")
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"The directory '{output_dir}' has been created.")

    # For each frame with mask annotations (video_segments)
        # Load image, draw masks, export
    for n, img_file in enumerate(tqdm(sam2.frame_names, desc="Exporting Debug Masks")):
        # print(f"{n} - {img_file}")
        image = Image.open(os.path.join(root_folder,img_file)).convert("RGB")
        masks = sam2.video_segments.get(n,{})
        overlay = sam2.overlay_masks(image, masks,verbose=False)
        overlay.save(os.path.join(output_dir,img_file)) 
        # print(f"Saved image {n}:{img_file}, Masks:{len(masks.keys())}")