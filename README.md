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

## Install
### ComfyUI Manager
After this repo is published to the Comfy Registry, install it from the ComfyUI Manager by searching for `Robe Nodes`.

### Manual install
1. Clone this repository into your `ComfyUI/custom_nodes` directory
2. Restart ComfyUI

## Registry metadata
This repo includes a `pyproject.toml` and a GitHub Actions workflow for publishing to the Comfy Registry, which powers the current ComfyUI Manager UI.

Before the first publish:
1. Create a publisher on the Comfy Registry
2. Confirm that the node id in `pyproject.toml` is the one you want to keep
3. Add a GitHub repository secret named `REGISTRY_ACCESS_TOKEN`
4. Bump the version in `pyproject.toml` before each release

## Dependencies
Runtime dependencies are listed in [requirements.txt](requirements.txt). Core packages provided by ComfyUI itself are not repeated there.

## License
This project is licensed under the MIT License.
