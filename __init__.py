import os

class VideoListNode:
    """
    A node that lists video files in a specified directory and cycles through them based on an index.

    Class methods
    -------------
    INPUT_TYPES (dict): 
        Defines the input parameters for the node.

    Attributes
    ----------
    RETURN_TYPES (`tuple`): 
        The type of each element in the output tuple.
    FUNCTION (`str`):
        The name of the entry-point method.
    CATEGORY (`str`):
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

        return (videos, videos[index])  # Return both the list and the current video

    def list_videos(self, directory):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        return [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in video_extensions]


# A dictionary that contains all nodes you want to export with their names
NODE_CLASS_MAPPINGS = {
    "VideoListNode": VideoListNode
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoListNode": "Video List Node"
}