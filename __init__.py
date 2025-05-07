"""
Custom nodes for the Comfy UI stable diffusion client.
"""

import os
import glob
import random

import ast
from PIL import Image, ImageOps
import numpy as np
import torch


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
                "model_type": (["ALL", "FLUX1", "HUN1", "SD15", "SDXL"], {
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



# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "List Video Path 🐤": ListVideoPath,
    "List Image Path 🐤": ListImagePath,
    "List Model Path 🐤": ListModelPath,
    "Indices Generator 🐤": IndicesGenerator,
    "Peaks Weights Generator 🐤": PeaksWeightsGenerator,
    "Image Input Switch 🐤": Image_Input_Switch,
    "Boolean Primitive 🐤": BooleanPrimitive,
}
