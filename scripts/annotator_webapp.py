import gradio as gr
import argparse
import os
import json
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
import torch
from collections import OrderedDict
import matplotlib.pyplot as plt
from sam2_handler import SAM2_Handler
import sam2_handler as s2h
from data_export import *
from utils import find_frames

# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# DRAW FUNCTIONS
# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def overlay_mask_on_image(sam2, image: Image.Image, masks: np.ndarray, labels, alpha=0.75):
    """
    Draw one or more semi-transparent segmentation masks on a PIL image.
    Supports:
      - 2D masks (H, W)
      - 3D masks (N, H, W)
    Each mask gets its own distinct color.
    """
    # Convert input image to RGBA
    base = image.convert("RGBA")

    # Ensure masks is numpy array
    masks = np.array(masks)
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]  # (H, W) -> (1, H, W)
    elif masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)  # (N,1,H,W) -> (N,H,W)

    num_masks, H, W = masks.shape

    # Use a tab20 colormap for up to 20 distinct colors (loop if >20)
    # cmap = plt.get_cmap("tab20")
    cmap = plt.get_cmap("hsv") # also tried "turbo"
    for i, label in zip(range(masks.shape[0]),labels):
        color_id = sam2.labels.index(label)
        mask = masks[i].astype(bool)
        if not mask.any():
            continue

        # color = (np.array(cmap(color_id % 20)[:3]) * 255).astype(np.uint8)
        color = (np.array(cmap(color_id / len(sam2.labels))[:3]) * 255).astype(np.uint8)
        overlay = Image.new("RGBA", base.size, tuple(color) + (int(alpha * 255),))
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))#, mode="L")

        # Use alpha_composite correctly by premasking the overlay
        overlay.putalpha(mask_img.point(lambda p: p * alpha))
        base = Image.alpha_composite(base, overlay)

    return base.convert("RGB")


def draw_points_on_image(sam2, image: Image.Image, current_idx, marker_radius=8):
    """
    Draw positive (green) and negative (red) points on a PIL image.
    """
    H, W, C = image.shape
    print(f"Input Size: {image.shape}")
    pil_img = Image.fromarray(image)
    pil_img.save("/workspace/data/test.png")
    print(f"Image: {type(pil_img)}")
    draw = ImageDraw.Draw(pil_img)
    print("Prompts for current image:\nID - LBL - COORDS")
    for obj_id, frames_dict in sam2.inference_state["point_inputs_per_obj"].items():
        if current_idx in frames_dict.keys():
            # Draw points for this object
            points = frames_dict[current_idx]["point_coords"].detach().cpu().numpy()
            labels = frames_dict[current_idx]["point_labels"].detach().cpu().numpy()
            # Remove any batch dimensions but *keep* point structure
            points = np.reshape(points, (-1, 2))   # Guarantees shape (N, 2)
            labels = np.reshape(labels, (-1,))     # Guarantees shape (N,)
            print(type(labels))
            if type(labels) is int: labels = [labels]
            print(f"Points: {points}")
            print(f"Labels: {labels}")
            # if not points:
            #     continue
            for p, l in zip(points,labels):
                color = "green" if l == 1 else "red"
                # Coordinates need to be rescaled from internal model size to image size 
                x = (float(p[0])/1024)*W
                y = (float(p[1])/1024)*H

                print(f"{obj_id} - {l} - {p} -> [{x}, {y}]")

                draw.line((x - marker_radius -1, y-1, x + marker_radius -1, y-1), fill="white", width=2)
                draw.line((x-1, y - marker_radius-1, x-1, y + marker_radius -1), fill="white", width=2)
                draw.line((x - marker_radius, y, x + marker_radius, y), fill=color, width=2)
                draw.line((x, y - marker_radius, x, y + marker_radius), fill=color, width=2)
    
    # pil_img.save("/workspace/data/test2.png")
    return np.array(pil_img)

# TODO: Implement this for visualizing bounding boxes
# def draw_box_on_image(image: Image.Image, box, color="green", width=3):
#     """
#     Draw a bounding box on a PIL image.
#     """
#     draw = ImageDraw.Draw(image)
#     x0, y0, x1, y1 = box
#     draw.rectangle([x0, y0, x1, y1], outline=color, width=width)
#     return image


# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Image sequence management
# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# def list_images(folder):
#     valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
#     files = sorted([f for f in os.listdir(folder) if f.lower().endswith(valid_exts)])
#     return [os.path.join(folder, f) for f in files]

def get_frames(search_folder, search_regex):
    frames, frame_range = find_frames(search_folder,search_regex)
    minf, maxf = frame_range
    status = f"Found {len(frames)} frames [{minf}-{maxf}]"
    return status, minf, maxf

def update_folder(sam2, folder, search_regex, min_frame, max_frame):
    if not os.path.isdir(folder):
        return None, "Invalid folder path.", [], 0
    # image_paths, frame_range= find_frames(folder,search_regex,min_frame,max_frame)
    # LOAD IMAGES INTO SAM
    image_paths, frame_range = sam2.set_video_folder(folder,search_regex,min_frame,max_frame)
    if not image_paths:
        return None, "No images found in folder.", [], 0
    image, status = sam2.load_image(0, image_paths)
    return image, status, image_paths, 0, frame_range[0], frame_range[1]

def next_image(sam2, current_idx, image_paths):
    if not image_paths:
        return None, "No images loaded.", current_idx
    new_idx = (current_idx + 1) % len(image_paths)  # min(current_idx + 1, len(image_paths) - 1)
    image, status = sam2.load_image(new_idx, image_paths)
    return image, status, new_idx

def jump_to_image(sam2, current_idx, jump_idx, image_paths):
    if not image_paths:
        return None, "No images loaded.", current_idx
    new_idx = max(0,min(int(jump_idx), len(image_paths) - 1)) 
    image, status = sam2.load_image(new_idx, image_paths)
    return image, status, new_idx

def prev_image(sam2, current_idx, image_paths):
    if not image_paths:
        return None, "No images loaded.", current_idx
    new_idx = (current_idx - 1) % len(image_paths)  # max(current_idx - 1, 0)
    image, status = sam2.load_image(new_idx, image_paths)
    return image, status, new_idx


def handle_click(sam2, evt: gr.SelectData, image_paths, current_idx, label, click_log, negative_prompt):
    ''' Add prompt to the model, update, and redraw masks for current image. '''
    # global model, inference_state, original_image
    if not image_paths:
        return "No image loaded.", click_log
    x, y = evt.index
    filename = os.path.basename(image_paths[current_idx])
    entry = {"file": filename, "x": x, "y": y, "label": label}
    click_log.append(entry)
    status = f"Added prompt for {label} at ({x}, {y}) on {filename}"

    # Check inference state to see if object already exists
    if label in sam2.inference_state['obj_ids']:
        label_idx = sam2.inference_state['obj_id_to_idx'].get(label)
        print(f"Preexisting object - Internal ID: {label_idx}")
    else:
        print(f"Adding new object - No previous data to gather")

    points = np.array([[x, y]], dtype=np.float32)
    if not negative_prompt:
        plabels = np.array([1], np.int32) 
    else:
        plabels = np.array([0], np.int32) 
    print(f"Adding prompt: FRAME_ID={current_idx}, OBJ_ID={label}, PTS={points}, LBL={plabels}")

    _, out_obj_ids, out_mask_logits = sam2.model.add_new_points_or_box(
        inference_state=sam2.inference_state,
        frame_idx=current_idx,
        obj_id=label,
        points=points,
        labels=plabels,
        clear_old_points=False
    )
    print(f"Out OBJ IDs: {out_obj_ids}")
    print(f"Out logits shape: {out_mask_logits.shape}")
    # show_points(points, plabels, image)

    sam2.inspect_inference_state()

    draw_mask = (out_mask_logits > 0.0).cpu().numpy()
    # If you ever get multiple masks (e.g. multiple objects at once), you can pick one explicitly:
    # draw_mask = (out_mask_logits[0, 0] > 0).cpu().numpy()
    image = sam2.original_image.copy()
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    out_image = overlay_mask_on_image(sam2, image,draw_mask,out_obj_ids)

    return status, click_log, out_image, sam2



# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# GUI Definition
# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def build_ui(sam2_handler):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr.Markdown("## SAM2 Annotation Tool\nSelect a folder of images to load into SAM2, then click to add prompts.")
            with gr.Column():
                with gr.Row():
                    folder_box = gr.Textbox(value=sam2_handler.data_folder, label="Frames Path (Local to Docker)") 
                    regex_box = gr.Textbox(value=sam2_handler.frame_regex, label="Frame search regex")
                search_btn = gr.Button("Find Frames")
            with gr.Column():
                with gr.Row():
                    minrng_box = gr.Number(label="First Frame", value="0", interactive=True)
                    maxrng_box = gr.Number(label="Last Frame", value="-1", interactive=True)
                select_btn = gr.Button("Load Frames Into SAM2")

        output_image = gr.Image(label="Segmentation Overlay", show_label=False)
        with gr.Row():
            status_text = gr.Textbox(label="Status", interactive=False)
            label_dropdown = gr.Dropdown(
                choices=sam2_handler.labels, 
                label="Select Label",
                value=sam2_handler.labels[0]
            )
            with gr.Column():
                prompt_type_box = gr.Checkbox(label="Negative Prompt", value=False)
                # bb_box = gr.Checkbox(label="Draw Bounding Boxes", value=True)
                with gr.Row():
                    reset_image_btn = gr.Button("Remove all prompts for this image")
                    show_prompts_btn = gr.Button("Draw current input prompts")

        with gr.Row():
            with gr.Column():
                with gr.Row():
                    prev_btn = gr.Button("⬅️ Previous")
                    next_btn = gr.Button("Next ➡️")
                jump_btn = gr.Button("Jump")
            with gr.Column():
                jump_text = gr.Number(label="Jump to:", interactive=True)

            with gr.Column():
                propagate_btn = gr.Button("Propagate All From Here")
                propagate_n_btn = gr.Button("Propagate N From Here")
            with gr.Column():
                propagate_n_text = gr.Number(label="N=", value="5", interactive=True)
        
        with gr.Row():
            reset_sam_btn = gr.Button("Reset SAM")
            clear_frames_btn = gr.Button("Clear Inference Masks")
            print_inf_btn = gr.Button("Print Inference State")

        with gr.Row():
            export_ann_btn = gr.Button("Export YOLO Annotations")
            export_mask_btn = gr.Button("Export Binary Masks")
            export_debugmask_btn = gr.Button("Export Colored Masks")

        # hidden state elements
        image_paths_state = gr.State([])
        current_idx_state = gr.State(0)
        click_log_state = gr.State([])
        sam2_state = gr.State(sam2_handler)


        # === BUTTON CALLBACKS ====
        reset_image_btn.click(s2h.w_reset_image_prompts, #
                        inputs=[sam2_state, current_idx_state, image_paths_state],
                        outputs=[output_image, sam2_state])
        show_prompts_btn.click(draw_points_on_image, #
                        inputs=[sam2_state, output_image, current_idx_state],
                        outputs=[output_image])
        search_btn.click(get_frames, # Search for frames in folder matching regex
                        inputs=[folder_box, regex_box], 
                        outputs=[status_text, minrng_box, maxrng_box])
        select_btn.click(update_folder, # Load Frames within Range to SAM2
                        inputs=[sam2_state, folder_box, regex_box, minrng_box, maxrng_box], 
                        outputs=[output_image, status_text, image_paths_state, current_idx_state, minrng_box, maxrng_box])
        next_btn.click(next_image, #
                        inputs=[sam2_state, current_idx_state, image_paths_state],
                        outputs=[output_image, status_text, current_idx_state])
        jump_btn.click(jump_to_image, #
                        inputs=[sam2_state, current_idx_state, jump_text, image_paths_state],
                        outputs=[output_image, status_text, current_idx_state])
        propagate_btn.click(s2h.w_propagate_frames, #
                        inputs=[sam2_state, current_idx_state, image_paths_state],
                        outputs=[output_image, sam2_state])
        propagate_n_btn.click(s2h.w_propagate_frames, #
                        inputs=[sam2_state,current_idx_state, image_paths_state, propagate_n_text],
                        outputs=[output_image, sam2_state])
        prev_btn.click(prev_image, #
                        inputs=[sam2_state, current_idx_state, image_paths_state],
                        outputs=[output_image, status_text, current_idx_state])
        reset_sam_btn.click(s2h.w_reset_sam,#
                        inputs=[sam2_state, current_idx_state, image_paths_state],
                        outputs=[output_image, status_text, current_idx_state, sam2_state])
        clear_frames_btn.click(s2h.w_clear_inference_frames,#
                        inputs=[sam2_state, current_idx_state, image_paths_state],
                        outputs=[output_image, status_text, current_idx_state, sam2_state])
        print_inf_btn.click(s2h.w_inspect_inference_state,#
                        inputs=[sam2_state],
                        outputs=[])
        
        # Export buttons
        export_ann_btn.click(export_yolo_annotations,
                        inputs=[sam2_state, folder_box,folder_box],
                        outputs=[])
        export_mask_btn.click(export_binary_masks,
                        inputs=[sam2_state, folder_box],
                        outputs=[])
        export_debugmask_btn.click(export_debug_masks,
                        inputs=[sam2_state, folder_box],
                        outputs=[])

        # Handle image click
        output_image.select(
            fn=handle_click,
            inputs=[sam2_state, image_paths_state, current_idx_state, label_dropdown, click_log_state,prompt_type_box],
            outputs=[status_text, click_log_state, output_image, sam2_state]
        )

        # Auto-load folder if provided at launch
        demo.load(update_folder, 
                inputs=[sam2_state, folder_box, regex_box, minrng_box, maxrng_box], 
                outputs=[output_image, status_text, image_paths_state, current_idx_state, minrng_box, maxrng_box],
                trigger_mode="once")

    return demo


# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# Entry Point / MAIN
# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="/workspace/data/mini", help="Folder containing image sequence")
    parser.add_argument("--config", type=str, default="/workspace/config/annotator_config.json", help="JSON config file")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    # Load Json config 
    print(f"Loading Config: {args.config}")
    global config
    with open(args.config, "r") as f:
        config = json.load(f)
    print(json.dumps(config, indent=4))

    # Load SAM2 Model
    sam2 = SAM2_Handler(config)

    # sam2.set_video_folder(config["data_folder"],config["frame_regex"],0,-1)
    # exit()
    # sam2 = load_sam2_model(args.data)


    demo = build_ui(sam2)
    # demo.update_folder(args.data)
    demo.launch(server_name="0.0.0.0", server_port=args.port)
