# Comfy UI Robe Nodes
This is a collection of utility nodes for the ComfyUI stable diffusion client that provides enhanced file path handling capabilities.

## Description
The package includes:

- **List Video Path Node** 🐤: A node that lists and manages video files in a specified directory. It supports:
  - Multiple video formats (.mp4, .avi, .mov, .mkv)
  - Cyclic index selection
  - Returns file paths and count information

- **List Image Path Node** 🐤: A node that handles image files in a directory with features like:
  - Support for common image formats (.jpg, .jpeg, .png, .gif)
  - Cyclic index selection
  - Returns file paths, dimensions, count, and image tensor data
  - Automatic EXIF orientation handling
  - RGB conversion

## Getting Started
1. Clone or download this repository
2. Import into the custom_nodes directory of your ComfyUI custom node directory
3. Restart ComfyUI to load the new nodes

## Dependencies
- PIL (Pillow)
- numpy
- torch

## License
This project is licensed under the MIT License.