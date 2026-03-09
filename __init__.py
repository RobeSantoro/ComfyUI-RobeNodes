"""
Custom nodes for the Comfy UI stable diffusion client.
V3 Schema Implementation
"""

import os
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
from comfy_api.latest import ComfyExtension, io, ui

# Suppress Gemini IMAGE_SAFETY warnings
warnings.filterwarnings("ignore", message="IMAGE_SAFETY is not a valid FinishReason")

# Set up logging for GeminiBanana
logging.basicConfig(level=logging.INFO)
_gemini_logger = logging.getLogger("GeminiBanana")

# Suppress verbose HTTP logging from Google API client
for _logger_name in ["httpx", "httpcore", "google.genai", "google.auth", "urllib3", "requests"]:
    logging.getLogger(_logger_name).setLevel(logging.ERROR)


# Custom types
AnyType = io.Custom("*")
FloatList = io.Custom("FLOAT")


class ListVideoPath(io.ComfyNode):
    """
    List the video files full path in a directory
    and output the selected video path by specified index
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="List Video Path 🐤",
            display_name="List Video Path 🐤",
            category="RobeNodes",
            description="List all video files in the specified directory and output the selected video path, count and video name/path.",
            inputs=[
                io.String.Input(
                    "directory",
                    default="W:\\ATTRAVERSO\\gammaPre\\masks_portrait_3.4",
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                ),
                io.Combo.Input(
                    "cycle",
                    options=["enable", "disable"],
                    default="enable",
                ),
            ],
            outputs=[
                io.Custom("LIST").Output(display_name="video_paths_list"),
                io.String.Output(display_name="selected_video_path"),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def _list_videos(cls, directory):
        videos = []
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

        if not os.path.exists(directory):
            return []
        
        for filename in os.listdir(directory):
            if os.path.splitext(filename)[1].lower() in video_extensions:
                videos.append(filename)
        
        return videos

    @classmethod
    def execute(cls, directory, index, cycle) -> io.NodeOutput:
        videos = cls._list_videos(directory)
        if not videos:
            return io.NodeOutput([], None, 0)

        if cycle == "enable":
            index = index % len(videos)
        else:
            index = min(index, len(videos) - 1)

        selected_video = os.path.join(directory, videos[index])
        return io.NodeOutput(videos, selected_video, len(videos))


class ListImagePath(io.ComfyNode):
    """
    List the image files path in a directory and output the selected
    image path, count, dimensions and image tensor
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="List Image Path 🐤",
            display_name="List Image Path 🐤",
            category="RobeNodes",
            description="List all image files in the specified directory and output the selected image path, count, dimensions and image tensor.",
            inputs=[
                io.String.Input(
                    "directory",
                    default="W:\\ATTRAVERSO\\OPERE\\scrape\\images_renamed\\Selection",
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                ),
                io.Combo.Input(
                    "cycle",
                    options=["enable", "disable"],
                    default="enable",
                ),
            ],
            outputs=[
                io.Custom("LIST").Output(display_name="image_paths_list"),
                io.String.Output(display_name="selected_image_path"),
                io.Int.Output(display_name="count"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def _list_images(cls, directory):
        image_files = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']

        if not os.path.exists(directory):
            return []
        
        for filename in os.listdir(directory):
            if os.path.splitext(filename)[1].lower() in image_extensions:
                image_files.append(filename)
        
        return image_files

    @classmethod
    def execute(cls, directory, index, cycle) -> io.NodeOutput:
        images = cls._list_images(directory)
        if not images:
            return io.NodeOutput([], None, 0, 0, 0, None)

        if cycle == "enable":
            index = index % len(images)
        else:
            index = min(index, len(images) - 1)

        selected_image = os.path.join(directory, images[index])
        
        width = 0
        height = 0
        image_tensor = None
        try:
            with Image.open(selected_image) as img:
                img = ImageOps.exif_transpose(img)
                img = img.convert("RGB")
                width, height = img.size
                image = np.array(img).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image)[None,]
        except Exception:
            image_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        return io.NodeOutput(images, selected_image, len(images), width, height, image_tensor)


class ListModelPath(io.ComfyNode):
    """
    List all *.safetensors, *.ckpt, *.pt and *.pth files in the specified directory
    and output the selected model path, count and model tensor
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="List Model Path 🐤",
            display_name="List Model Path 🐤",
            category="RobeNodes",
            description="List all *.safetensors files in the specified directory and output the selected model path, count and model name/path",
            inputs=[
                io.String.Input(
                    "directory",
                    default="E:\\MODELS\\checkpoints",
                ),
                io.Combo.Input(
                    "model_type",
                    options=["ALL", "FLUX1", "HUN1", "SD15", "LCM", "SDXL"],
                    default="ALL",
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                ),
                io.Combo.Input(
                    "cycle",
                    options=["enable", "disable"],
                    default="enable",
                ),
            ],
            outputs=[
                io.String.Output(display_name="model_paths_list"),
                AnyType.Output(display_name="selected_model_path"),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def _list_models(cls, directory, model_type="ALL"):
        models = []
        model_extensions = ['.safetensors', '.ckpt', '.pth', '.pt']

        if not os.path.exists(directory):
            return []
        
        for root, _, files in os.walk(directory):
            for filename in files:
                if os.path.splitext(filename)[1].lower() in model_extensions:
                    rel_path = os.path.relpath(root, directory)
                    if rel_path == '.':
                        models.append(filename)
                    else:
                        models.append(os.path.join(rel_path, filename))

        if model_type != "ALL":
            models = [m for m in models if m.startswith(model_type + "\\")]
        
        return models

    @classmethod
    def execute(cls, directory, model_type, index, cycle) -> io.NodeOutput:
        models = cls._list_models(directory, model_type)
        if not models:
            return io.NodeOutput([], "", 0)

        if cycle == "enable":
            index = index % len(models)
        else:
            index = min(index, len(models) - 1)

        selected_model = models[index]
        model_paths_list = "\n".join([f"[{i}] - {model}" for i, model in enumerate(models)])
        
        return io.NodeOutput(model_paths_list, selected_model, len(models))


class IndicesGenerator(io.ComfyNode):
    """
    Divides the frames count by the number of images and
    returns a comma separated string of equally spaced indices.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Indices Generator 🐤",
            display_name="Indices Generator 🐤",
            category="RobeNodes",
            description="Returns a comma separated string of equally spaced indices from the total number of frames and the number of images.",
            inputs=[
                io.Int.Input(
                    "frames_count",
                    default=0,
                    min=0,
                ),
                io.Int.Input(
                    "images_count",
                    default=0,
                    min=0,
                ),
            ],
            outputs=[
                io.String.Output(display_name="indices"),
            ],
        )

    @classmethod
    def execute(cls, frames_count, images_count) -> io.NodeOutput:
        indices = [i for i in range(0, frames_count, round(frames_count / (images_count)))]

        if len(indices) > images_count:
            indices = indices[:images_count]
        
        indices_str = ", ".join(map(str, indices))
        
        return io.NodeOutput(indices_str)


class PeaksWeightsGenerator(io.ComfyNode):
    """
    Generates a list of weights from a binary string to be used with the
    "Generate Peaks Weights" node from Yvann Nodes
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Peaks Weights Generator 🐤",
            display_name="Peaks Weights Generator 🐤",
            category="RobeNodes",
            description="Generates a list of weights from a binary string.",
            inputs=[
                io.Int.Input(
                    "frames_count",
                    default=0,
                    min=0,
                ),
                io.String.Input(
                    "one_indexes",
                    default="0, 4, 8, 12",
                ),
                io.Boolean.Input(
                    "specify_peaks_manually",
                    default=False,
                ),
                io.String.Input(
                    "peaks_binary_string",
                    multiline=True,
                    default="[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]",
                ),
            ],
            outputs=[
                FloatList.Output(display_name="peaks_weights"),
            ],
        )

    @classmethod
    def execute(cls, frames_count, one_indexes, specify_peaks_manually, peaks_binary_string) -> io.NodeOutput:
        if specify_peaks_manually:
            try:
                peaks_binary = ast.literal_eval(peaks_binary_string)
                
                if not all(x in (0, 1) for x in peaks_binary):
                    raise ValueError("All values must be either 0 or 1")
                    
                return io.NodeOutput(peaks_binary)
                
            except Exception as e:
                print(f"Error converting peaks binary string: {e}")
                return io.NodeOutput([])
        else:
            peaks_weights = [0] * frames_count
            try:
                for index in one_indexes.split(","):
                    index = int(index.strip())
                    if 0 <= index < frames_count:
                        peaks_weights[index] = 1
                return io.NodeOutput(peaks_weights)
            except Exception as e:
                print(f"Error generating peaks weights: {e}")
                return io.NodeOutput([])


class ImageInputSwitch(io.ComfyNode):
    """
    Switch between two images based on a boolean input.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Input Switch 🐤",
            display_name="Image Input Switch 🐤",
            category="RobeNodes",
            description="Switch between two images based on a boolean input.",
            inputs=[
                io.Image.Input("image_a"),
                io.Image.Input("image_b"),
                io.Boolean.Input("boolean", force_input=True),
            ],
            outputs=[
                io.Image.Output(),
            ],
        )

    @classmethod
    def execute(cls, image_a, image_b, boolean) -> io.NodeOutput:
        if boolean:
            return io.NodeOutput(image_a)
        else:
            return io.NodeOutput(image_b)


class LatentInputSwitch(io.ComfyNode):
    """
    Switch between two latents based on a boolean input.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Latent Input Switch 🐤",
            display_name="Latent Input Switch 🐤",
            category="RobeNodes",
            description="Switch between two latents based on a boolean input.",
            inputs=[
                io.Latent.Input("latent_a"),
                io.Latent.Input("latent_b"),
                io.Boolean.Input("boolean", force_input=True),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def execute(cls, latent_a, latent_b, boolean) -> io.NodeOutput:
        if boolean:
            return io.NodeOutput(latent_a)
        else:
            return io.NodeOutput(latent_b)


class BooleanPrimitive(io.ComfyNode):
    """
    Primitive node to convert a boolean value to a string and vice versa.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Boolean Primitive 🐤",
            display_name="Boolean Primitive 🐤",
            category="RobeNodes",
            description="Primitive node to convert a boolean value to a string.",
            inputs=[
                io.Boolean.Input("value", default=False),
                io.Boolean.Input("reverse", default=False),
            ],
            outputs=[
                io.Boolean.Output(),
                io.String.Output(),
            ],
        )

    @classmethod
    def execute(cls, value: bool, reverse: bool) -> io.NodeOutput:
        if reverse:
            value = not value

        return io.NodeOutput(value, str(value))


class AudioWeightsToFadeMask(io.ComfyNode):
    """
    Converts audio weights (a single float or a list of floats)
    into a multiline FadeMask string format.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AudioWeights to FadeMask 🐤",
            display_name="AudioWeights to FadeMask 🐤",
            category="RobeNodes",
            description="Converts audio weights (float or list of floats) to FadeMask string: index:(value),",
            inputs=[
                FloatList.Input("audio_weights", force_input=True),
            ],
            outputs=[
                io.String.Output(display_name="fade_mask_string"),
            ],
        )

    @classmethod
    def execute(cls, audio_weights) -> io.NodeOutput:
        weights_list = []
        if isinstance(audio_weights, (int, float)):
            weights_list = [float(audio_weights)]
        elif isinstance(audio_weights, list):
            weights_list = audio_weights
        else:
            print(f"[AudioWeights_To_FadeMask] Warning: audio_weights is of unexpected type {type(audio_weights)}. Expected float or list. Returning empty string.")
            return io.NodeOutput("")

        if not weights_list:
            return io.NodeOutput("")

        fade_mask_lines = []
        for i, weight_item in enumerate(weights_list):
            try:
                weight_val = float(weight_item)
                fade_mask_lines.append(f"{i}:({weight_val}),")
            except (ValueError, TypeError):
                print(f"[AudioWeights_To_FadeMask] Warning: Could not convert item '{weight_item}' at index {i} to float. Skipping.")
                continue 
        
        result_string = "\n".join(fade_mask_lines)
        
        return io.NodeOutput(result_string)


def _easy_save(images, filename_prefix, output_type, prompt=None, extra_pnginfo=None):
    """Save or Preview Image helper function"""
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


class LoadImageBase64(io.ComfyNode):
    """
    Load Image from Base64 String
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="easy loadImageBase64",
            display_name="Load Image Base64",
            category="RobeNodes",
            description="Load an image from a base64 encoded string.",
            is_output_node=True,
            inputs=[
                io.String.Input("base64_data", default=""),
                io.Combo.Input(
                    "image_output",
                    options=["Hide", "Preview", "Save", "Hide/Save"],
                    default="Preview",
                ),
                io.String.Input("save_prefix", default="ComfyUI"),
            ],
            outputs=[
                io.Image.Output(),
                io.Mask.Output(),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def _convert_color(cls, image):
        if len(image.shape) > 2 and image.shape[2] >= 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @classmethod
    def execute(cls, base64_data, image_output, save_prefix) -> io.NodeOutput:
        prompt = cls.hidden.prompt
        extra_pnginfo = cls.hidden.extra_pnginfo
        
        nparr = np.frombuffer(base64.b64decode(base64_data), np.uint8)

        result = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        channels = cv2.split(result)
        if len(channels) > 3:
            mask = channels[3].astype(np.float32) / 255.0
            mask = torch.from_numpy(mask)
        else:
            mask = torch.ones(channels[0].shape, dtype=torch.float32, device="cpu")

        result = cls._convert_color(result)
        result = result.astype(np.float32) / 255.0
        new_images = torch.from_numpy(result)[None,]

        results = _easy_save(new_images, save_prefix, image_output, None, None)
        mask = mask.unsqueeze(0)

        if image_output in ("Hide", "Hide/Save"):
            return io.NodeOutput(new_images, mask)

        return io.NodeOutput(new_images, mask, ui={"images": results})


class SaveImageJPEG(io.ComfyNode):
    """
    Save images as JPEG files with configurable quality settings.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Save Image (JPEG) 🐤",
            display_name="Save Image (JPEG) 🐤",
            category="RobeNodes",
            description="Save images as JPEG files with configurable quality settings.",
            is_output_node=True,
            inputs=[
                io.Image.Input("images", tooltip="The images to save."),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip="The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes.",
                ),
                io.Combo.Input(
                    "quality",
                    options=["100", "95", "90", "85", "80", "75", "70", "60", "50"],
                    default="95",
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(cls, images, filename_prefix, quality) -> io.NodeOutput:
        quality = int(quality)
        output_dir = folder_paths.get_output_directory()
        
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, output_dir, images[0].shape[1], images[0].shape[0]
        )
        results = list()

        for batch_number, image in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img = img.convert("RGB")

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.jpg"
            img.save(os.path.join(full_output_folder, file), format="JPEG", quality=quality, subsampling=0)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": "output"
            })
            counter += 1

        return io.NodeOutput(ui={"images": results})


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


class GeminiBanana(io.ComfyNode):
    """
    Minimal Gemini API node for image generation and analysis.
    Supports gemini-2.5-flash-image and gemini-3-pro-image-preview models.
    """

    MODELS = [
        "gemini-2.5-flash-image",
        "gemini-3-pro-image-preview",
    ]

    ASPECT_RATIOS = {
        "none": (1024, 1024),
        "1:1": (1024, 1024),
        "16:9": (1408, 768),
        "9:16": (768, 1408),
        "4:3": (1280, 896),
        "3:4": (896, 1280),
    }

    _api_key = os.environ.get("GEMINI_API_KEY", "")
    _genai_available = None

    @classmethod
    def _check_genai(cls):
        if cls._genai_available is None:
            try:
                from google import genai
                cls._genai_available = True
            except ImportError:
                _gemini_logger.error("google-genai not installed. Run: pip install google-genai")
                cls._genai_available = False
        return cls._genai_available

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="GeminiBanana 🍌",
            display_name="GeminiBanana 🍌",
            category="RobeNodes",
            description="Generate images or analyze content using Google Gemini API (Nano Banana models)",
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                ),
                io.Combo.Input(
                    "operation_mode",
                    options=["generate_images", "analysis"],
                    default="generate_images",
                ),
                io.Combo.Input(
                    "model_name",
                    options=cls.MODELS,
                    default=cls.MODELS[0],
                ),
                io.Float.Input(
                    "temperature",
                    default=0.8,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                ),
                io.Image.Input("images", optional=True),
                io.String.Input("api_key", default="", optional=True),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFF, optional=True),
                io.Int.Input("batch_count", default=1, min=1, max=10, optional=True),
                io.Combo.Input(
                    "aspect_ratio",
                    options=["none", "1:1", "16:9", "9:16", "4:3", "3:4"],
                    default="none",
                    optional=True,
                ),
                io.Int.Input("max_images", default=6, min=1, max=16, optional=True),
                io.Int.Input("max_output_tokens", default=8192, min=1, max=32768, optional=True),
                io.Float.Input("api_call_delay", default=1.0, min=0.0, max=60.0, step=0.1, optional=True),
            ],
            outputs=[
                io.String.Output(display_name="text"),
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt,
        operation_mode,
        model_name,
        temperature,
        images=None,
        api_key="",
        seed=0,
        batch_count=1,
        aspect_ratio="1:1",
        max_images=6,
        max_output_tokens=8192,
        api_call_delay=1.0,
    ) -> io.NodeOutput:
        """Main generation function"""
        
        if not cls._check_genai():
            return io.NodeOutput("ERROR: google-genai not installed. Run: pip install google-genai", _gemini_create_placeholder())

        effective_key = api_key.strip() if api_key else cls._api_key
        if not effective_key:
            return io.NodeOutput("ERROR: No Gemini API key. Set GEMINI_API_KEY environment variable or provide in node.", _gemini_create_placeholder())

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=effective_key)

            if operation_mode == "generate_images":
                return cls._generate_images(
                    client, types, prompt, model_name, temperature, images,
                    seed, batch_count, aspect_ratio, max_images, api_call_delay
                )
            else:
                return cls._analyze_content(
                    client, types, prompt, model_name, temperature, images,
                    seed, max_images, max_output_tokens
                )

        except Exception as e:
            error_msg = str(e)
            _gemini_logger.error(f"Gemini API error: {error_msg}")
            if len(error_msg) > 500:
                error_msg = error_msg[:500] + "... [truncated]"
            return io.NodeOutput(f"ERROR: {error_msg}", _gemini_create_placeholder())

    @classmethod
    def _generate_images(
        cls, client, types, prompt, model_name, temperature, images,
        seed, batch_count, aspect_ratio, max_images, api_call_delay
    ) -> io.NodeOutput:
        """Generate images using Gemini"""
        
        target_width, target_height = cls.ASPECT_RATIOS.get(aspect_ratio, (1024, 1024))
        _gemini_logger.info(f"Generating {batch_count} images at {target_width}x{target_height}")

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

                response = client.models.generate_content(
                    model=model_name,
                    contents=content,
                    config=gen_config,
                )

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

        if all_images_bytes:
            try:
                pil_images = []
                for img_bytes in all_images_bytes:
                    pil_img = Image.open(BytesIO(img_bytes)).convert('RGB')
                    pil_images.append(pil_img)

                if not pil_images:
                    return io.NodeOutput(f"Failed to process images.\n\n{status}", _gemini_create_placeholder())

                first_w, first_h = pil_images[0].size
                for i in range(1, len(pil_images)):
                    if pil_images[i].size != (first_w, first_h):
                        pil_images[i] = pil_images[i].resize((first_w, first_h), Image.LANCZOS)

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

                return io.NodeOutput(result_text, image_tensor)

            except Exception as proc_err:
                _gemini_logger.error(f"Error processing images: {proc_err}")
                return io.NodeOutput(f"Error processing images: {proc_err}\n\n{status}", _gemini_create_placeholder())
        else:
            return io.NodeOutput(f"No images generated with {model_name}.\n\n{status}", _gemini_create_placeholder())

    @classmethod
    def _analyze_content(
        cls, client, types, prompt, model_name, temperature, images,
        seed, max_images, max_output_tokens
    ) -> io.NodeOutput:
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

        if images is not None and isinstance(images, torch.Tensor) and images.nelement() > 0:
            ref_images = _gemini_prepare_batch_images(images, max_images, 1568)
            if ref_images:
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
            return io.NodeOutput(response.text, _gemini_create_placeholder())
        except Exception as e:
            _gemini_logger.error(f"Analysis error: {e}")
            return io.NodeOutput(f"ERROR: {str(e)}", _gemini_create_placeholder())


# ============================================================================
# V3 Extension Registration
# ============================================================================

class RobeNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ListVideoPath,
            ListImagePath,
            ListModelPath,
            IndicesGenerator,
            PeaksWeightsGenerator,
            ImageInputSwitch,
            LatentInputSwitch,
            BooleanPrimitive,
            AudioWeightsToFadeMask,
            LoadImageBase64,
            SaveImageJPEG,
            GeminiBanana,
        ]


async def comfy_entrypoint() -> RobeNodesExtension:
    return RobeNodesExtension()
