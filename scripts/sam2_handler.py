from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
import os
from os.path import join

from collections import OrderedDict

import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw
import torch
import numpy as np

# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
# FUNCTIONS FROM SAM2 TO MODIFY IMAGE LOADING BEHAVIOR
# ------------------------------|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
def _load_img_as_tensor(img_path, image_size):
    img_pil = Image.open(img_path)
    img_np = np.array(img_pil.convert("RGB").resize((image_size, image_size)))
    if img_np.dtype == np.uint8:  # np.uint8 is expected for JPEG images
        img_np = img_np / 255.0
    else:
        raise RuntimeError(f"Unknown image dtype: {img_np.dtype} on {img_path}")
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_width, video_height = img_pil.size  # the original video size
    return img, video_height, video_width

def load_video_frames_from_scan(video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda")
    ):
    '''Expects images named like {str}_{frame_num}.jpg'''
    if isinstance(video_path, str) and os.path.isdir(video_path):
        image_folder = video_path
    else:
        raise NotImplementedError("Video path is not a string and directory.")   
    file_types = [".jpg", ".jpeg", ".JPG", ".JPEG"]
    frame_names = [
        p
        for p in os.listdir(image_folder)
        if os.path.splitext(p)[-1] in file_types
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[1]))
    
    num_frames = len(frame_names)

    print(f"Found {num_frames} frames.")
    if num_frames == 0:
        raise RuntimeError(f"no {file_types} images found in {image_folder}")
    img_paths = [os.path.join(image_folder, frame_name) for frame_name in frame_names]
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    # if async_loading_frames:
    #     lazy_images = AsyncVideoFrameLoader(
    #         img_paths,
    #         image_size,
    #         offload_video_to_cpu,
    #         img_mean,
    #         img_std,
    #         compute_device,
    #     )
    #     return lazy_images, lazy_images.video_height, lazy_images.video_width

    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
    for n, img_path in enumerate(tqdm(img_paths, desc="frame loading (JPEG)")):
        images[n], video_height, video_width = _load_img_as_tensor(img_path, image_size)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width

@torch.inference_mode()
def init_state(
    model,
    video_path,
    offload_video_to_cpu=False,
    offload_state_to_cpu=False,
    async_loading_frames=False,
    ):
    """Initialize an inference state."""
    compute_device = model.device  # device of the model
    images, video_height, video_width = load_video_frames_from_scan(
        video_path=video_path,
        image_size=model.image_size,
        offload_video_to_cpu=offload_video_to_cpu,
        async_loading_frames=async_loading_frames,
        compute_device=compute_device,
    )
    inference_state = {}
    inference_state["images"] = images
    inference_state["num_frames"] = len(images)
    # whether to offload the video frames to CPU memory
    # turning on this option saves the GPU memory with only a very small overhead
    inference_state["offload_video_to_cpu"] = offload_video_to_cpu
    # whether to offload the inference state to CPU memory
    # turning on this option saves the GPU memory at the cost of a lower tracking fps
    # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
    # and from 24 to 21 when tracking two objects)
    inference_state["offload_state_to_cpu"] = offload_state_to_cpu
    # the original video height and width, used for resizing final output scores
    inference_state["video_height"] = video_height
    inference_state["video_width"] = video_width
    inference_state["device"] = compute_device
    if offload_state_to_cpu:
        inference_state["storage_device"] = torch.device("cpu")
    else:
        inference_state["storage_device"] = compute_device
    # inputs on each frame
    inference_state["point_inputs_per_obj"] = {}
    inference_state["mask_inputs_per_obj"] = {}
    # visual features on a small number of recently visited frames for quick interactions
    inference_state["cached_features"] = {}
    # values that don't change across frames (so we only need to hold one copy of them)
    inference_state["constants"] = {}
    # mapping between client-side object id and model-side object index
    inference_state["obj_id_to_idx"] = OrderedDict()
    inference_state["obj_idx_to_id"] = OrderedDict()
    inference_state["obj_ids"] = []
    # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
    inference_state["output_dict_per_obj"] = {}
    # A temporary storage to hold new outputs when user interact with a frame
    # to add clicks or mask (it's merged into "output_dict" before propagation starts)
    inference_state["temp_output_dict_per_obj"] = {}
    # Frames that already holds consolidated outputs from click or mask inputs
    # (we directly use their consolidated outputs during tracking)
    # metadata for each tracking frame (e.g. which direction it's tracked)
    inference_state["frames_tracked_per_obj"] = {}
    # Warm up the visual backbone and cache the image feature on frame 0
    model._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    return inference_state

class SAM2_Handler:
    def __init__(self,config_dict, device="cuda"):
        print("=== Loading SAM2 Model ===")
        self.device = device
        print(f"Device: {self.device}")
        self.sam2_location = config_dict["sam2_location"]
        print(f"Location: {self.sam2_location}")
        self.checkpoint = join(self.sam2_location,config_dict["sam2_checkpoint"])
        print(f"Checkpoint: {self.checkpoint}")
        self.model_cfg = config_dict["sam2_config"]
        print(f"Config: {self.model_cfg}")
        
        self.model = build_sam2_video_predictor(self.model_cfg, 
                                                self.checkpoint, 
                                                device=self.device)
        
        # [!!] Necessary for correcting frames
        self.model.add_all_frames_to_correct_as_cond = True 
        print(f"add_all_frames_to_correct_as_cond: {self.model.add_all_frames_to_correct_as_cond}")

        self.inference_state = None
        self.video_segments = {} # Stores propagated mask data
        self.original_image = None
        self.frame_names = []
        self.labels = config_dict["object_labels"]

        print("=== MODEL LOADED ===", device)

    def set_sam_video_folder(self,frame_dir):
        inference_state = init_state(self.model,video_path=frame_dir)
        self.model.reset_state(self.inference_state)
        self.frame_names = [
            p for p in os.listdir(frame_dir)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0].split('_')[1]))

    def reset_sam(self, current_idx, image_paths):
        # global model, inference_state, video_segments
        self.model.reset_state(self.inference_state)
        self.video_segments = {}
        image, status = self.load_image(current_idx, image_paths)
        print("SAM State Reset")
        return image, status, current_idx

    def reset_image_prompts(self, current_idx, image_paths):
        # global inference_state, video_segments
        for obj_id, frames_dict in self.inference_state["point_inputs_per_obj"].items():
            if current_idx in frames_dict.keys():
                del frames_dict[current_idx]
                print(f"Removed prompt for {self.inference_state['obj_idx_to_id'][obj_id]}")


        image, status = self.load_image(current_idx, image_paths)
        return image

    def clear_inference_frames(self, current_idx, image_paths):
        # global inference_state, video_segments
        self.video_segments = {}
        image, status = self.load_image(current_idx, image_paths)
        print("Inference Frames Cleared")
        return image, status, current_idx

    def propagate_frames(self, current_idx, image_paths,n_frames=None):
        # global inference_state, video_segments, model
        start_frame = current_idx 
        if n_frames is not None:
            try:
                n_frames = int(n_frames)
            except:
                print("ERROR: Couldn't convert N to an int...")
                return
        print(f"Propagating {'All' if n_frames is None else n_frames} Frames from frame {start_frame}...")

        # run propagation throughout the video and collect the results in a dict
        # OLD: Clear all frames 
        # video_segments = {}  # video_segments contains the per-frame segmentation results
        # NEW: Only clear frames after the current frame, leave previous frames untouched.
        keys_to_remove = [k for k in self.video_segments.keys() if k >= start_frame]
        for k in keys_to_remove:
            del self.video_segments[k]

        for out_frame_idx, out_obj_ids, out_mask_logits in self.model.propagate_in_video(self.inference_state,start_frame_idx=start_frame,max_frame_num_to_track=n_frames):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

        print(f"Vid Seg Keys: {self.video_segments.keys()}")
        print(f"Seg 0: {self.video_segments.get(0,None)}")

        image, status = self.load_image(current_idx, image_paths)

        print("Propagation Finished.")
        return image

    def inspect_inference_state(self):
        ''' Print out data inside current inference_state'''
        print(f"Inference State Type: {type(self.inference_state)}, Len: {len(self.inference_state.keys())}")
        for k, v in self.inference_state.items():
            if k not in ['images','temp_output_dict_per_obj','cached_features','output_dict_per_obj','constants','frames_tracked_per_obj']:
            # if False:
                print(f"> [ {k} ]: {v}")
            elif k in ['output_dict_per_obj','temp_output_dict_per_obj']:
                print(f"> [ {k} ]: Keys: {v.keys()}")
            else:
                print(f"> [ {k} ]: Type: {type(v)}")
    
    def load_image(self, idx, image_paths):
        # global original_image
        if not image_paths:
            return None, "No images found."
        idx = max(0, min(idx, len(image_paths)-1))
        image = Image.open(image_paths[idx]).convert("RGB")
        self.original_image = image.copy()
        masks = self.video_segments.get(idx,{})
        print(f"Loading image IDX:{idx}, Masks:{len(masks.keys())}")
        #TODO: IF LEN MASKS == 0: CHECK INFERENCE STATE FOR MASKS TO DRAW ON CURRENT FRAME
        #masks = fake_sam2_segmentation(image)
        overlay = self.overlay_masks(image, masks)
        status = f"Image {idx+1}/{len(image_paths)}: {os.path.basename(image_paths[idx])}"
        return overlay, status
    
    def overlay_masks(self, image, masks, alpha=0.75, verbose=True):
        '''
        Called when loading an image, gets masks from video_segments (only after propagating frames)
        '''
        # Use a tab20 colormap for up to 20 distinct colors (loop if >20)
        # cmap = plt.get_cmap("tab20")
        cmap = plt.get_cmap("hsv")
        base = image.convert("RGBA")
        for obj_name, m in masks.items():
            num_masks, H, W = m.shape
            obj_id = self.labels.index(obj_name)
            if verbose: print(f"Object Name: {obj_name} - ID: {obj_id}")

            for i in range(m.shape[0]):
                mask = m[i].astype(bool)
                if not mask.any(): # Skip empty masks
                    continue

            # color = (np.array(cmap(obj_id % 20)[:3]) * 255).astype(np.uint8)
            color = (np.array(cmap(obj_id / len(self.labels))[:3]) * 255).astype(np.uint8)
            overlay = Image.new("RGBA", base.size, tuple(color) + (int(alpha * 255),))
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))#, mode="L")

            # Use alpha_composite correctly by premasking the overlay
            overlay.putalpha(mask_img.point(lambda p: p * alpha))
            base = Image.alpha_composite(base, overlay)

        return base.convert("RGB")