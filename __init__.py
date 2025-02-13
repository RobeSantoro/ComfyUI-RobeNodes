"""
Custom nodes for the Comfy UI stable diffusion client.
"""

import os
import json
from PIL import Image, ImageOps
import numpy as np
import torch


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
    List all *.safetensors files in the specified directory
    and output the selected model path, count and model tensor
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "E:\\MODELS\\checkpoints",
                }),
                "include_subdirectories": (["enable", "disable"], {
                    "default": "disable",
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
    RETURN_NAMES = ("model_paths_list", "selected_model_path", "count")
    FUNCTION = "execute"
    CATEGORY = "RobeNodes"

    def list_models(self, directory, include_subdirectories):
        models = []
        model_extensions = ['.safetensors']

        if not os.path.exists(directory):
            return []
        
        if include_subdirectories == "enable":
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
        else:
            # Only search in the root directory
            for filename in os.listdir(directory):
                if os.path.splitext(filename)[1].lower() in model_extensions:
                    models.append(filename)
        
        return models

    def execute(self, directory, include_subdirectories, index, cycle):
        models = self.list_models(directory, include_subdirectories)
        if not models:
            return ([], None, 0)  # No models found

        # Cycle logic
        if cycle == "enable":
            index = index % len(models)  # Wrap around using modulo
        else:
            index = min(index, len(models) - 1)  # Limit to max index

        selected_model = models[index]
        
        # Return the list, full path of the current model, and count
        return (models, selected_model, len(models))


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "List Video Path 🐤": ListVideoPath,
    "List Image Path 🐤": ListImagePath,
    "List Model Path 🐤": ListModelPath
}
