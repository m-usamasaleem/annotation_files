# WebDataset Format for PyTorch

This repository utilizes the WebDataset format for efficient data loading in PyTorch. WebDataset is particularly useful for handling large datasets by storing them in sharded TAR archives and leveraging efficient data streaming and processing capabilities.

## Data Preparation

### 1. Collecting Data

Ensure all data is collected and organized in a suitable format (e.g., `.npz`, `.png`, etc.) before creating WebDataset archives.

### 2. Creating `.data.pyd` Files

#### Requirements for Annotation

Annotations should include the following fields:
- `keypoints_2d`: Array of 44 keypoints, each represented as a 3-dimensional vector.
- `keypoints_3d`: Array of 44 keypoints, each represented as a 4-dimensional vector.
- `center`, `scale`: Two values each, specifying the center and scale of the person in the image.
- `body_pose`: Array of 72 values representing body pose information.
- `has_body_pose`: Boolean (1 or 0) indicating whether body pose information is available.
- `has_betas`: Boolean (1 or 0) indicating whether betas (body shape parameters) are available.
- `betas`: Array of 10 values representing body shape parameters.
- `personid`: Identifier (e.g., `0`) uniquely identifying each person in the dataset.

### Image Format

Images can be in any format such as JPEG (`.jpg`), PNG (`.png`), etc.

#### Data Type

Ensure all data is stored with `numpy.float32` type wherever applicable for compatibility with PyTorch.

### 3. Sorting Data

When creating the `.tar` archive, ensure data is sorted using the `--sort=name` option to maintain consistent ordering across shards.

Example command for creating `.tar` archive:
```bash
tar --sort=name -cf data.tar /path/to/source/folder
```

For further details on sorting TAR files, refer to [YouTube tutorials](https://www.youtube.com/watch?v=v_PacO-3OGQ) and practical examples on GitHub.

## Example

![Example Image](https://github.com/user-attachments/assets/bcb8fedb-98f9-4f0f-ae20-6db00d508ef1)

This image illustrates the structure and process of organizing data into WebDataset format.

## Resources

- WebDataset GitHub Repository: [webdataset/webdataset](https://github.com/webdataset/webdataset)
- YouTube Tutorial on Sorting TAR Files: [Sorting TAR Files Tutorial](https://www.youtube.com/watch?v=v_PacO-3OGQ)

By following these steps and guidelines, you can effectively prepare and utilize datasets in WebDataset format for your PyTorch applications.
