import os
import time
import traceback
import runpod
import numpy as np
import torch
from PIL import Image

# Local imports
from segmind_utils import get_error_response, gpu_metrics, endpoint_id, gpu_id
from utilities import upload_to_s3, bucket_name
import utils_api
from segmentation import segment_image
import helpers

# Initialize logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("---- GPU INFO ----")
print(f"GPU Endpoint: {endpoint_id}")
print(f"GPU ID: {gpu_id}")
print(f"GPU Metrics: {gpu_metrics}")

# Global variable for lazy loading
# The segmentation module handles lazy loading internally, but we can warm it up if needed
# For now, we rely on the internal manager in segmentation.py

def verify_params(points_per_side=None, max_masks=None, threshold=None, 
                  pred_iou_thresh=None, image_quality=None, type="serverless"):
    """
    Validate input parameters are within acceptable ranges.
    Returns error response if invalid, None otherwise.
    """
    if points_per_side is not None:
        if not isinstance(points_per_side, int) or not (0 <= points_per_side <= 128):
            return get_error_response("points_per_side must be an integer between 0 and 128", 400, type=type)
            
    if max_masks is not None:
        if not isinstance(max_masks, int) or not (0 <= max_masks <= 100):
             return get_error_response("max_masks must be an integer between 0 and 100", 400, type=type)
             
    if threshold is not None:
        if not isinstance(threshold, (float, int)) or not (0.0 <= float(threshold) <= 1.0):
            return get_error_response("threshold must be a float between 0.0 and 1.0", 400, type=type)
            
    if pred_iou_thresh is not None:
        if not isinstance(pred_iou_thresh, (float, int)) or not (0.0 <= float(pred_iou_thresh) <= 1.0):
            return get_error_response("pred_iou_thresh must be a float between 0.0 and 1.0", 400, type=type)

    if image_quality is not None:
        if not isinstance(image_quality, int) or not (1 <= image_quality <= 100):
            return get_error_response("image_quality must be an integer between 1 and 100", 400, type=type)
            
    return None

@torch.inference_mode()
def handler(job):
    start_time = time.time()
    job_input = job["input"]
    
    # Extract request ID
    request_id = job_input.get("request_id", "request_id")
    
    try:
        request_prefix = f"[{request_id}] [{gpu_id}] [{endpoint_id}]"
        filtered_body = {
            key: value for key, value in job_input.items()
            if key not in ("image", "points_input", "point_labels_input", "boxes_input")
        }
        print(f"{request_prefix} JSON Body:{filtered_body}")
        print(f"{request_prefix} API Request start time:{start_time}")
        
        logger.info(f"Received request: {request_id}")
        
        # --- 1. Parse Input Parameters ---
        
        # Image input (Required)
        image_input = job_input.get("image")
        if not image_input:
            return get_error_response("Input 'image' is required", 400, type="serverless")
        
        # Prompts
        text_prompt = job_input.get("text_prompt")
        points_input_str = job_input.get("points_input")
        point_labels_input_str = job_input.get("point_labels_input", "[[1]]")
        boxes_input_str = job_input.get("boxes_input")
        
        # Parse structured inputs using utils_api
        try:
            input_points = utils_api.format_points(points_input_str)
            input_boxes = utils_api.format_boxes(boxes_input_str)
            
            input_labels = None
            if input_points:
                num_points = len(input_points[0][0]) if input_points else 0
                input_labels = utils_api.format_labels(point_labels_input_str, num_points)
        except ValueError as e:
             return get_error_response(f"Input parsing error: {str(e)}", 400, type="serverless")

        # Settings
        max_masks = job_input.get("max_masks", 10)
        if max_masks is not None:
            max_masks = int(max_masks)
            
        # Hardcoded settings as per requirements
        mode = None # Auto-detect
        normalize_boxes = False # Pixel coordinates
        mask_threshold = 0.5
        image_format = "jpeg"
        
        threshold = float(job_input.get("threshold", 0.5))
        points_per_side = int(job_input.get("points_per_side", 32))
        pred_iou_thresh = float(job_input.get("pred_iou_thresh", 0.88))
        
        # Hardcoded advanced settings
        multimask_output = False
        binarize_masks = True
        
        # Output Control
        return_preview = job_input.get("return_preview", True)
        return_overlay = job_input.get("return_overlay", False)
        return_masks = job_input.get("return_masks", False)
        output_format = job_input.get("output_format", "image").lower() # "image" or "rle"
        
        image_quality = 95

        # --- 1.5 Verify Parameters ---
        validation_error = verify_params(
            points_per_side=points_per_side,
            max_masks=max_masks,
            threshold=threshold,
            pred_iou_thresh=pred_iou_thresh,
            image_quality=image_quality,
            type="serverless"
        )
        if validation_error:
            return validation_error

        # --- 2. Run Inference ---
        
        infer_start = time.time()
        print(f"{request_prefix} Pre-processing Time:{infer_start - start_time}")
        
        # Call segment_image directly
        try:
            # Note: segment_image expects 'image' as one of [str, Image.Image, np.ndarray]
            # utils_api.base64_to_image handles base64 strings if it's a raw base64 string
            # but helpers.load_image_from_source handles urls and base64 too.
            # We will pass the raw image_input and let load_image_from_source handle it, 
            # unless it's a base64 string that helpers might miss (e.g. missing data uri prefix handled by helpers? yes)
            
            result = segment_image(
                image=image_input,
                text_prompt=text_prompt if text_prompt else None,
                input_points=input_points,
                input_labels=input_labels,
                input_boxes=input_boxes,
                max_masks=max_masks if max_masks and max_masks > 0 else None,
                threshold=threshold,
                mask_threshold=mask_threshold,
                normalize_boxes=normalize_boxes,
                return_preview=False, # We generate previews manually to control format/upload
                return_overlay=False, # We generate overlays manually
                return_masks=True,    # Always get masks to process them
                multimask_output=multimask_output,
                binarize_masks=binarize_masks,
                points_per_side=points_per_side if points_per_side > 0 else None,
                pred_iou_thresh=pred_iou_thresh,
                mode=mode
            )
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}")
            logger.error(traceback.format_exc())
            return get_error_response(f"Segmentation failed: {str(e)}", 500, type="serverless")

        infer_end = time.time()
        infer_time = infer_end - infer_start
        print(f"{request_prefix} Generation Time:{infer_time}")
        
        # --- 3. Process Outputs & Upload to S3 ---
        
        # Metadata headers
        original_size = result.get("original_size", (0,0))
        img_height, img_width = original_size
        print(f"{request_prefix} img_width:{img_width}, img_height:{img_height}")
        meta_data = {
            "height": original_size[0], 
            "width": original_size[1], 
            "num_objects": result.get("num_objects", 0)
        }
        
        response_data = {
            "status": "Success",
            "infer_time": infer_time,
            "outputs": result.get("num_objects", 0)
        }
        
        raw_masks = result.get("masks") # numpy array [N, H, W]
        
        # Helper to save and upload
        def save_and_upload(img, suffix):
            filename = f"{request_id}_{suffix}.{image_format}"
            if image_format == "jpeg":
                 img.save(filename, format="JPEG", quality=image_quality)
            elif image_format == "png":
                 img.save(filename, format="PNG")
            elif image_format == "webp":
                 img.save(filename, format="WEBP", quality=image_quality)
            
            object_key = f"sam3/{request_id}/{filename}"
            upload_to_s3(filename, object_key)
            
            # Cleanup
            if os.path.exists(filename):
                os.remove(filename)
                
            return f"https://{bucket_name}.s3.amazonaws.com/{object_key}"

        # 1. Preview Mask (Composite)
        if return_preview and raw_masks is not None and len(raw_masks) > 0:
            try:
                # Merge masks
                merged_mask = helpers.merge_masks(raw_masks, method="union")
                # Create binary image (white on black)
                preview_img = Image.fromarray((merged_mask * 255).astype(np.uint8), mode='L')
                
                preview_url = save_and_upload(preview_img, "preview_mask")
                response_data["preview_mask"] = preview_url
                
                # If only preview was requested, set as 'image'
                if not return_overlay and not return_masks:
                    response_data["image"] = preview_url
                    response_data["image_format"] = image_format
            except Exception as e:
                logger.error(f"Failed to create preview: {e}")
                return get_error_response("Failed to upload preview output", 400, type="serverless")

        # 2. Overlay Image
        if return_overlay and raw_masks is not None and len(raw_masks) > 0:
            try:
                # We need the original image for overlay
                # Load it again or pass it if we had it. segment_image loads it internally but doesn't return the PIL object
                # We can reload it efficiently
                pil_image = helpers.load_image_from_source(image_input)
                
                overlay_img = helpers.create_overlay(pil_image, raw_masks)
                
                overlay_url = save_and_upload(overlay_img, "overlay")
                response_data["overlay_image"] = overlay_url
                
                # If only overlay was requested (and not preview/masks), set as 'image' and drop overlay_image key
                if not return_preview and not return_masks:
                    response_data["image"] = overlay_url
                    response_data["image_format"] = image_format
                    response_data.pop("overlay_image", None)
            except Exception as e:
                logger.error(f"Failed to create overlay: {e}")
                return get_error_response("Failed to upload overlay output", 400, type="serverless")

        # 3. Individual Masks
        if return_masks and raw_masks is not None and len(raw_masks) > 0:
            try:
                if output_format == "rle":
                    rles = helpers.masks_to_rle(raw_masks)
                    response_data["masks"] = rles
                    response_data["output_format"] = "rle"
                else:
                    # Image format
                    mask_urls = []
                    for i, mask in enumerate(raw_masks):
                        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
                        mask_url = save_and_upload(mask_img, f"mask_{i}")
                        mask_urls.append(mask_url)
                    response_data["masks"] = mask_urls
                
                response_data["masks_count"] = len(raw_masks)
            except Exception as e:
                 logger.error(f"Failed to process individual masks: {e}")
                 return get_error_response("Failed to upload mask outputs", 400, type="serverless")

        # Remove generic 'outputs' count when multiple output assets are included
        output_variants = sum([
            1 if "preview_mask" in response_data else 0,
            1 if "overlay_image" in response_data else 0,
            0 if "masks" in response_data else 0
        ])

        # Keep 'outputs' only when exactly one display image is returned.
        if output_variants != 1:
            response_data.pop("outputs", None)

        # Add boxes and scores if available
        if "boxes" in result:
            response_data["boxes"] = result["boxes"]
        if "scores" in result:
            response_data["scores"] = result["scores"]

        # Headers
        headers = {
            "X-generation-time": infer_time,
            "X-output-metadata": meta_data,
            "x-gpu-info": gpu_metrics
        }
        
        # Wrap response
        print(f"{request_prefix} Post-processing Time:{time.time()-infer_end}")
        print(f"{request_prefix} API Response Time:{time.time()}")
        return [response_data, headers]

    except Exception as e:
        logger.error(f"Handler failed: {e}")
        logger.error(traceback.format_exc())
        print(f"{request_prefix} Error:{e}")
        return get_error_response(str(e), 500, type="serverless")

# Start the handler
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
