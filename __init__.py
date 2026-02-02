"""
Custom nodes for the Comfy UI stable diffusion client.
"""

import os
import glob
import random
import time
import logging
import warnings

import ast
from PIL import Image, ImageOps
import numpy as np
import torch
import cv2
import base64
from io import BytesIO
import folder_paths

# Suppress Gemini IMAGE_SAFETY warnings
warnings.filterwarnings("ignore", message="IMAGE_SAFETY is not a valid FinishReason")

# Set up logging for GeminiBanana
logging.basicConfig(level=logging.INFO)
_gemini_logger = logging.getLogger("GeminiBanana")

# Suppress verbose HTTP logging from Google API client
for _logger_name in ["httpx", "httpcore", "google.genai", "google.auth", "urllib3", "requests"]:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


class AnyType(str):
    """A special type that can be connected to any other types. Credit to pythongosssss"""

    def __ne__(self, __value: object) -> bool:
        return False


any_type = AnyType("*")


class ListVideoPath:
    """
    List the video files full path in a directory
    and output the selected video path by specified index
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "W:\\ATTRAVERSO\\gammaPre\\masks_portrait_3.4",
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                }),
                "cycle": (["enable", "disable"], {
                    "default": "enable",
                }),
            },
        }

    RETURN_TYPES = ("LIST", "STRING", "INT")
    RETURN_NAMES = ("video_paths_list", "selected_video_path", "count")
    FUNCTION = "execute"
    CATEGORY = "RobeNodes"
    DESCRIPTION = "List all video files in the specified directory and output the selected video path, count and video name/path."

    def list_videos(self, directory):
        videos = []
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

        if not os.path.exists(directory):
            return []
        
        for filename in os.listdir(directory):
            if os.path.splitext(filename)[1].lower() in video_extensions:
                videos.append(filename)
        
        return videos

    def execute(self, directory, index, cycle):

        videos = self.list_videos(directory)
        if not videos:
            return ([], None, 0)  # No videos found

        # Cycle logic
        if cycle == "enable":
            index = index % len(videos)  # Wrap around using modulo
        else:
            index = min(index, len(videos) - 1)  # Limit to max index

        selected_video = os.path.join(directory, videos[index])

        # Return the list, full path of the current video, and count
        return (videos, selected_video, len(videos))


class ListImagePath:
    """
    List the image files path in a directory and output the selected
    image path, count, dimensions and image tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    # Change this to your desired default directory
                    "default": "W:\\ATTRAVERSO\\OPERE\\scrape\\images_renamed\\Selection",
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                }),
                "cycle": (["enable", "disable"], {
                    "default": "enable",
                }),
            },
        }

    RETURN_TYPES = ("LIST", "STRING", "INT", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("image_paths_list", "selected_image_path", "count", "width", "height", "image")
    FUNCTION = "execute"
    CATEGORY = "RobeNodes"
    DESCRIPTION = "List all image files in the specified directory and output the selected image path, count, dimensions and image tensor."

    def execute(self, directory, index, cycle):
        images = self.list_images(directory)
        if not images:
            return ([], None, 0, 0, 0, None)  # No images found

        # Cycle logic
        if cycle == "enable":
            index = index % len(images)  # Wrap around using modulo
        else:
            index = min(index, len(images) - 1)  # Limit to max index

        selected_image = os.path.join(directory, images[index])
        
        # Get image dimensions and load image
        width = 0
        height = 0
        image_tensor = None
        try:
            with Image.open(selected_image) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                width, height = img.size
                # Convert to tensor
                image = np.array(img).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image)[None,]
        except Exception:
            # Return empty tensor if image can't be loaded
            image_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        # Return the list, selected image path, count, dimensions and image tensor
        return (images, selected_image, len(images), width, height, image_tensor)

    def list_images(self, directory):
        image_files = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

        if not os.path.exists(directory):
            return []
        
        for filename in os.listdir(directory):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_files.append(filename)
        
        return image_files


class ListModelPath:
    """
    List all *.safetensors, *.ckpt, *.pt and *.pth files in the specified directory
    and output the selected model path, count and model tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "E:\\MODELS\\checkpoints",
                }),
                "model_type": (["ALL", "FLUX1", "HUN1", "SD15", "LCM", "SDXL"], {
                    "default": "ALL",
                }),
                "index": ("INT", {
                    "default": 0,
                    "min": 0,
                }),
                "cycle": (["enable", "disable"], {
                    "default": "enable",
                }),
            },
        }

    RETURN_TYPES = ("STRING", any_type, "INT")  
    RETURN_NAMES = ("model_paths_list", "selected_model_path", "count")
    FUNCTION = "execute"
    CATEGORY = "RobeNodes"
    DESCRIPTION = "List all *.safetensors files in the specified directory and output the selected model path, count and model name/path"

    def list_models(self, directory, model_type="ALL"):
        models = []
        model_extensions = ['.safetensors', '.ckpt', '.pth', '.pt']

        if not os.path.exists(directory):
            return []
        
        # Recursive search through subdirectories
        for root, _, files in os.walk(directory):
            for filename in files:
                if os.path.splitext(filename)[1].lower() in model_extensions:
                    # Get the relative path from the base directory
                    rel_path = os.path.relpath(root, directory)
                    if rel_path == '.':
                        # If file is in the root directory, just use filename
                        models.append(filename)
                    else:
                        # Otherwise combine subdirectory + filename
                        models.append(os.path.join(rel_path, filename))

        # Filter by model type if specified
        if model_type != "ALL":
            models = [m for m in models if m.startswith(model_type + "\\")]
        
        return models

    def execute(self, directory, model_type, index, cycle):
        models = self.list_models(directory, model_type)
        if not models:
            return ([], "", 0)  # No models found

        # Cycle logic
        if cycle == "enable":
            index = index % len(models)  # Wrap around using modulo
        else:
            index = min(index, len(models) - 1)  # Limit to max index

        selected_model = models[index]

        # The list of models as a string with their indices on a new line
        model_paths_list = "\n".join([f"[{i}] - {model}" for i, model in enumerate(models)])
        
        # Return the list, selected model as a string (will be converted to COMBO by any_type), and count
        return (model_paths_list, selected_model, len(models))


class IndicesGenerator:
    """
    Divides the frames count by the number of images and
    returns a comma separated string of equally spaced indices.
    For example, if frames_count is 250 and images_count is 4, it returns 4 indices: "0, 62, 124, 186".
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_count": ("INT", {
                    "default": 0,
                    "min": 0,
                }),
                "images_count": ("INT", {
                    "default": 0,
                    "min": 0,
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("indices",)
    FUNCTION = "execute"
    CATEGORY = "RobeNodes"
    DESCRIPTION = "Returns a comma separated string of equally spaced indices from the total number of frames and the number of images."

    def execute(self, frames_count, images_count):
        # Generate a number of indices equals to the image_count
        # and round them to the nearest integer
        indices = [i for i in range(0, frames_count, round(frames_count / (images_count)))]

        # If there are more indices than images_count, remove the latest ones
        if len(indices) > images_count:
            indices = indices[:images_count]
        
        # Convert indices to comma separated string
        indices_str = ", ".join(map(str, indices))
        
        return (indices_str,)


class PeaksWeightsGenerator:
    """
    Generates a list of weights from a binary string to be used with the
    "Generate Peaks Weights" node from Yvann Nodes
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames_count": ("INT", {
                    "default": 0,
                    "min": 0,
                }),
                "one_indexes": ("STRING", {
                    "default": "0, 4, 8, 12", # for a 16 frames sequence
                    "forceInput": False
                }),
                "specify_peaks_manually": ("BOOLEAN", {"default": False}),
                "peaks_binary_string": ("STRING", {
                    "multiline": True,
                    "default": "[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]", # for a 16 frames sequence
                    "forceInput": False
                })
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("peaks_weights",)
    FUNCTION = "generate_peaks"
    CATEGORY = "RobeNodes"
    DESCRIPTION = "Generates a list of weights from a binary string."

    def generate_peaks(self, peaks_binary_string, specify_peaks_manually, frames_count, one_indexes):
        if specify_peaks_manually:
            try:
                # Convert string representation of list to actual list
                peaks_binary = ast.literal_eval(peaks_binary_string)
                
                # Ensure all values are either 0 or 1
                if not all(x in (0, 1) for x in peaks_binary):
                    raise ValueError("All values must be either 0 or 1")
                    
                return (peaks_binary,)
                
            except Exception as e:
                print(f"Error converting peaks binary string: {e}")
                # Return a default empty list in case of error
                return ([],)
        else:
            # Generate the peaks weights from the frames count and one_indexes string list
            peaks_weights = [0] * frames_count
            try:
                for index in one_indexes.split(","):
                    index = int(index.strip())
                    if 0 <= index < frames_count:
                        peaks_weights[index] = 1
                return (peaks_weights,)
            except Exception as e:
                print(f"Error generating peaks weights: {e}")
                return ([],)


class Image_Input_Switch:
    """
    Switch between two images based on a boolean input. From WAS Suite/Logic
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "boolean": ("BOOLEAN", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_input_switch"

    CATEGORY = "RobeNodes"
    DESCRIPTION = "Switch between two images based on a boolean input."

    def image_input_switch(self, image_a, image_b, boolean=True):

        if boolean:
            return (image_a, )
        else:
            return (image_b, )


class BooleanPrimitive:
    """
    Primitive node to convert a boolean value to a string and vice versa. From Art Venture/Utils
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "value": ("BOOLEAN", {"default": False}),
                "reverse": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "STRING")
    CATEGORY = "RobeNodes"
    FUNCTION = "boolean_primitive"

    def boolean_primitive(self, value: bool, reverse: bool):
        if reverse:
            value = not value

        return (value, str(value))


class AudioWeights_To_FadeMask:
    """
    Converts audio weights (a single float or a list of floats)
    into a multiline FadeMask string format, where each line is '{index}:({value}),'.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_weights": ("FLOAT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fade_mask_string",)
    FUNCTION = "convert_to_fade_mask"
    CATEGORY = "RobeNodes"
    DESCRIPTION = "Converts audio weights (float or list of floats) to FadeMask string: index:(value),"

    def convert_to_fade_mask(self, audio_weights: any) -> tuple[str,]: # audio_weights can be float or list[float]
        weights_list = []
        if isinstance(audio_weights, (int, float)):
            weights_list = [float(audio_weights)]
        elif isinstance(audio_weights, list):
            weights_list = audio_weights
        else:
            print(f"[AudioWeights_To_FadeMask] Warning: audio_weights is of unexpected type {type(audio_weights)}. Expected float or list. Returning empty string.")
            return ("",)

        if not weights_list:
            return ("",)

        fade_mask_lines = []
        for i, weight_item in enumerate(weights_list):
            try:
                weight_val = float(weight_item) # Ensure weight is float for consistent formatting.
                fade_mask_lines.append(f"{i}:({weight_val}),")
            except (ValueError, TypeError):
                print(f"[AudioWeights_To_FadeMask] Warning: Could not convert item '{weight_item}' at index {i} to float. Skipping.")
                continue 
        
        result_string = "\n".join(fade_mask_lines)
        
        return (result_string,)


def easySave(images, filename_prefix, output_type, prompt=None, extra_pnginfo=None):
    """Save or Preview Image from https://github.com/yolain/ComfyUI-Easy-Use"""
    from nodes import PreviewImage, SaveImage
    if output_type in ["Hide", "None"]:
        return list()
    elif output_type in ["Preview", "Preview&Choose"]:
        filename_prefix = 'easyPreview'
        results = PreviewImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']
    else:
        results = SaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']


class loadImageBase64:
    """
    Load Image from Base64 String from https://github.com/yolain/ComfyUI-Easy-Use
    """
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base64_data": ("STRING", {"default": ""}),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
            },
            "optional": {

            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    OUTPUT_NODE = True
    FUNCTION = "load_image"
    CATEGORY = "RobeNodes"

    def convert_color(self, image,):
        if len(image.shape) > 2 and image.shape[2] >= 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_image(self, base64_data, image_output, save_prefix, prompt=None, extra_pnginfo=None):
        nparr = np.frombuffer(base64.b64decode(base64_data), np.uint8)

        result = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        channels = cv2.split(result)
        if len(channels) > 3:
            mask = channels[3].astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
        else:
            mask = torch.ones(channels[0].shape,
                              dtype=torch.float32, device="cpu")

        result = self.convert_color(result)
        result = result.astype(np.float32) / 255.0
        new_images = torch.from_numpy(result)[None,]

        results = easySave(new_images, save_prefix, image_output, None, None)
        mask = mask.unsqueeze(0)

        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                    "result": (new_images, mask)}

        return {"ui": {"images": results},
                "result": (new_images, mask)}


class SaveImageJPEG:
    """
    Save images as JPEG files with configurable quality settings.
    """

    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."}),
                "quality": ([100, 95, 90, 85, 80, 75, 70, 60, 50], {"default": 95}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "RobeNodes"
    DESCRIPTION = "Save images as JPEG files with configurable quality settings."

    def save_images(self, images, filename_prefix="ComfyUI", quality=95, prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
        )
        results = list()

        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Convert to RGB (JPEG doesn't support alpha channel)
            img = img.convert("RGB")

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.jpg"
            img.save(os.path.join(full_output_folder, file), format="JPEG", quality=quality, subsampling=0)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}


# ============================================================================
# GeminiBanana - Minimal Gemini API Node for Image Generation
# ============================================================================

def _gemini_tensor_to_pil(tensor):
    """Convert a ComfyUI tensor to PIL image"""
    try:
        tensor = tensor.cpu()
        if tensor.dim() == 4:
            tensor = tensor[0] if tensor.shape[0] >= 1 else tensor.squeeze(0)
        if tensor.dim() == 3:
            if tensor.shape[0] == 3 and tensor.shape[-1] != 3:
                tensor = tensor.permute(1, 2, 0)
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        numpy_array = np.clip(tensor.numpy() * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(numpy_array)
    except Exception as e:
        _gemini_logger.error(f"Error in tensor_to_pil: {e}")
        return Image.new('RGB', (512, 512), color=(99, 99, 99))


def _gemini_resize_image(image, max_size=1024):
    """Resize image while preserving aspect ratio"""
    width, height = image.size
    ratio = min(max_size / width, max_size / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    return image.resize((new_width, new_height), Image.LANCZOS)


def _gemini_prepare_batch_images(images, max_images=6, max_size=1024):
    """Process batch images for Gemini API"""
    prepared = []
    try:
        if images is None or (isinstance(images, torch.Tensor) and images.nelement() == 0):
            return []
        if isinstance(images, torch.Tensor) and images.dim() == 4:
            batch_size = min(images.shape[0], max_images)
            for i in range(batch_size):
                pil_img = _gemini_tensor_to_pil(images[i])
                pil_img = _gemini_resize_image(pil_img, max_size)
                prepared.append(pil_img)
        elif isinstance(images, torch.Tensor) and images.dim() == 3:
            pil_img = _gemini_tensor_to_pil(images)
            pil_img = _gemini_resize_image(pil_img, max_size)
            prepared.append(pil_img)
    except Exception as e:
        _gemini_logger.error(f"Error preparing batch images: {e}")
    return prepared


def _gemini_create_placeholder(width=1024, height=1024):
    """Create a placeholder image tensor"""
    img = Image.new('RGB', (width, height), color=(99, 99, 99))
    img_array = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_array)[None,]


class GeminiBanana:
    """
    Minimal Gemini API node for image generation and analysis.
    Supports gemini-2.5-flash-image and gemini-3-pro-image-preview models.
    """

    # Predefined model list
    MODELS = [
        "gemini-2.5-flash-image",
        "gemini-3-pro-image-preview",
    ]

    # Aspect ratio dimensions
    ASPECT_RATIOS = {
        "none": (1024, 1024),
        "1:1": (1024, 1024),
        "16:9": (1408, 768),
        "9:16": (768, 1408),
        "4:3": (1280, 896),
        "3:4": (896, 1280),
    }

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        self.genai_available = self._check_genai()
        if self.api_key:
            _gemini_logger.info("Gemini API key found in environment")

    def _check_genai(self):
        """Check if Google Generative AI SDK is available"""
        try:
            from google import genai
            return True
        except ImportError:
            _gemini_logger.error("google-genai not installed. Run: pip install google-genai")
            return False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "operation_mode": (
                    ["generate_images", "analysis"],
                    {"default": "generate_images"}
                ),
                "model_name": (cls.MODELS, {"default": cls.MODELS[0]}),
                "temperature": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": 10}),
                "aspect_ratio": (["none", "1:1", "16:9", "9:16", "4:3", "3:4"], {"default": "none"}),
                "max_images": ("INT", {"default": 6, "min": 1, "max": 16}),
                "max_output_tokens": ("INT", {"default": 8192, "min": 1, "max": 32768}),
                "api_call_delay": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("text", "image")
    FUNCTION = "generate"
    CATEGORY = "RobeNodes"
    DESCRIPTION = "Generate images or analyze content using Google Gemini API (Nano Banana models)"

    def generate(
        self,
        prompt,
        operation_mode="generate_images",
        model_name="gemini-2.5-flash-image",
        temperature=0.8,
        images=None,
        api_key="",
        seed=0,
        batch_count=1,
        aspect_ratio="1:1",
        max_images=6,
        max_output_tokens=8192,
        api_call_delay=1.0,
    ):
        """Main generation function"""
        
        if not self.genai_available:
            return ("ERROR: google-genai not installed. Run: pip install google-genai", _gemini_create_placeholder())

        # Resolve API key
        effective_key = api_key.strip() if api_key else self.api_key
        if not effective_key:
            return ("ERROR: No Gemini API key. Set GEMINI_API_KEY environment variable or provide in node.", _gemini_create_placeholder())

        try:
            from google import genai
            from google.genai import types

            # Create client
            client = genai.Client(api_key=effective_key)

            if operation_mode == "generate_images":
                return self._generate_images(
                    client, types, prompt, model_name, temperature, images,
                    seed, batch_count, aspect_ratio, max_images, api_call_delay
                )
            else:
                return self._analyze_content(
                    client, types, prompt, model_name, temperature, images,
                    seed, max_images, max_output_tokens
                )

        except Exception as e:
            error_msg = str(e)
            _gemini_logger.error(f"Gemini API error: {error_msg}")
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "... [truncated]"
            return (f"ERROR: {error_msg}", _gemini_create_placeholder())

    def _generate_images(
        self, client, types, prompt, model_name, temperature, images,
        seed, batch_count, aspect_ratio, max_images, api_call_delay
    ):
        """Generate images using Gemini"""
        
        target_width, target_height = self.ASPECT_RATIOS.get(aspect_ratio, (1024, 1024))
        _gemini_logger.info(f"Generating {batch_count} images at {target_width}x{target_height}")

        # Prepare reference images
        ref_images = []
        if images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
            ref_images = _gemini_prepare_batch_images(images, max_images, max(target_width, target_height))
            _gemini_logger.info(f"Prepared {len(ref_images)} reference images")

        all_images_bytes = []
        all_text = []
        status = ""

        for i in range(batch_count):
            if i > 0 and api_call_delay > 0:
                _gemini_logger.info(f"Waiting {api_call_delay:.1f}s before next API call...")
                time.sleep(api_call_delay)

            try:
                current_seed = (seed + i) % (2**31 - 1)
                _gemini_logger.info(f"Batch {i+1}/{batch_count}, seed: {current_seed}")

                # Build generation config
                gen_config = types.GenerateContentConfig(
                    temperature=temperature,
                    response_modalities=["Text", "Image"],
                    seed=current_seed,
                    safety_settings=[
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                    ],
                )

                # Build content
                if aspect_ratio != "none":
                    content_text = f"Generate a detailed, high-quality image with dimensions {target_width}x{target_height} of: {prompt}"
                    print(content_text)
                else:
                    content_text = f"{prompt}"
                    print(content_text)
                if ref_images:
                    content = [content_text] + ref_images
                else:
                    content = content_text

                # Call API
                response = client.models.generate_content(
                    model=model_name,
                    contents=content,
                    config=gen_config,
                )

                # Extract images and text from response
                batch_images = []
                batch_text = ""

                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    if "<!DOCTYPE" not in part.text and "<html" not in part.text.lower():
                                        batch_text += part.text + "\n"
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    try:
                                        batch_images.append(part.inline_data.data)
                                    except Exception as img_err:
                                        _gemini_logger.error(f"Error extracting image: {img_err}")

                if batch_images:
                    all_images_bytes.extend(batch_images)
                    status += f"Batch {i+1} (seed {current_seed}): {len(batch_images)} image(s)\n"
                    if batch_text.strip():
                        all_text.append(f"Batch {i+1}:\n{batch_text.strip()}")
                else:
                    status += f"Batch {i+1} (seed {current_seed}): No images generated\n"
                    if batch_text.strip():
                        all_text.append(f"Batch {i+1} (no image):\n{batch_text.strip()}")

            except Exception as batch_err:
                status += f"Batch {i+1} error: {str(batch_err)}\n"
                _gemini_logger.error(f"Batch {i+1} error: {batch_err}")

        # Convert all images to tensors
        if all_images_bytes:
            try:
                pil_images = []
                for img_bytes in all_images_bytes:
                    pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
                    pil_images.append(pil_img)

                if not pil_images:
                    return (f"Failed to process images.\n\n{status}", _gemini_create_placeholder())

                # Ensure consistent dimensions
                first_w, first_h = pil_images[0].size
                for i in range(1, len(pil_images)):
                    if pil_images[i].size != (first_w, first_h):
                        pil_images[i] = pil_images[i].resize((first_w, first_h), Image.LANCZOS)

                # Convert to tensors
                tensors = []
                for pil_img in pil_images:
                    img_array = np.array(pil_img).astype(np.float32) / 255.0
                    tensors.append(torch.from_numpy(img_array)[None,])

                image_tensor = torch.cat(tensors, dim=0)

                result_text = f"Generated {len(all_images_bytes)} images using {model_name}.\n"
                result_text += f"Prompt: {prompt}\nSeed: {seed}\nResolution: {first_w}x{first_h}\n"
                if all_text:
                    result_text += "\n----- Generated Text -----\n" + "\n\n".join(all_text)
                result_text += f"\n\n----- Status -----\n{status}"

                return (result_text, image_tensor)

            except Exception as proc_err:
                _gemini_logger.error(f"Error processing images: {proc_err}")
                return (f"Error processing images: {proc_err}\n\n{status}", _gemini_create_placeholder())
        else:
            return (f"No images generated with {model_name}.\n\n{status}", _gemini_create_placeholder())

    def _analyze_content(
        self, client, types, prompt, model_name, temperature, images,
        seed, max_images, max_output_tokens
    ):
        """Analyze images/content using Gemini"""

        gen_config = types.GenerateContentConfig(
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            seed=seed,
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ],
        )

        # Prepare content
        if images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
            ref_images = _gemini_prepare_batch_images(images, max_images, 1568)
            if ref_images:
                # Build multipart content
                parts = [{"text": prompt}]
                for img in ref_images:
                    img_bytes = BytesIO()
                    img.save(img_bytes, format='PNG')
                    parts.append({
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": img_bytes.getvalue()
                        }
                    })
                contents = [{"parts": parts}]
            else:
                contents = [{"text": prompt}]
        else:
            contents = [{"text": prompt}]

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=gen_config,
            )
            return (response.text, _gemini_create_placeholder())
        except Exception as e:
            _gemini_logger.error(f"Analysis error: {e}")
            return (f"ERROR: {str(e)}", _gemini_create_placeholder())


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "List Video Path 🐤": ListVideoPath,
    "List Image Path 🐤": ListImagePath,
    "List Model Path 🐤": ListModelPath,
    "Indices Generator 🐤": IndicesGenerator,
    "Peaks Weights Generator 🐤": PeaksWeightsGenerator,
    "Image Input Switch 🐤": Image_Input_Switch,
    "Boolean Primitive 🐤": BooleanPrimitive,
    "AudioWeights to FadeMask 🐤": AudioWeights_To_FadeMask,
    "easy loadImageBase64": loadImageBase64,
    "Save Image (JPEG) 🐤": SaveImageJPEG,
    "GeminiBanana 🍌": GeminiBanana,
}
