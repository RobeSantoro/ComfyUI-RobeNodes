import os


class ListVideoPath:
    """
    The category under which this node will appear in the UI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "E:/COMFY/videos",  # Change this to your desired default directory
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

    RETURN_TYPES = ("LIST", "STRING")
    FUNCTION = "execute"
    CATEGORY = "Robe"

    def execute(self, directory, index, cycle):
        videos = self.list_videos(directory)
        if not videos:
            return ([], None)  # No videos found

        # Cycle logic
        if cycle == "enable":
            index = index % len(videos)  # Wrap around using modulo
        else:
            index = min(index, len(videos) - 1)  # Limit to max index

        selected_video = os.path.join(directory, videos[index])
        # Return both the list and the full path of the current video
        return (videos, selected_video)

    def list_videos(self, directory):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        return [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in video_extensions]


class ListImagePath:
    """
    The category under which this node will appear in the UI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "E:/COMFY/images",  # Change this to your desired default directory
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

    RETURN_TYPES = ("LIST", "STRING")
    FUNCTION = "execute"
    CATEGORY = "Robe"

    def execute(self, directory, index, cycle):
        images = self.list_images(directory)
        if not images:
            return ([], None)  # No images found

        # Cycle logic
        if cycle == "enable":
            index = index % len(images)  # Wrap around using modulo
        else:
            index = min(index, len(images) - 1)  # Limit to max index

        selected_image = os.path.join(directory, images[index])
        # Return both the list and the full path of the current image
        return (images, selected_image)

    def list_images(self, directory):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
        return [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in image_extensions]


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "List Video Path": ListVideoPath,
    "List Image Path": ListImagePath
}
