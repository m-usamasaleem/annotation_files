# WebDataset Format for PyTorch

This repository adopts the WebDataset format to optimize data loading in PyTorch, offering efficient handling of large datasets through sharded TAR archives and enhanced data streaming capabilities.

## Data Preparation

### 1. Data Collection

Ensure all data is gathered and structured in a suitable format (e.g., `.npz`, `.png`, etc.) before creating WebDataset archives.

### 2. Creation of `.data.pyd` Files

#### Annotation Requirements

Annotations must include the following fields:
- `keypoints_2d`: Array of 44 keypoints, each represented as a 3-dimensional vector.
- `keypoints_3d`: Array of 44 keypoints, each represented as a 4-dimensional vector.
- `center`, `scale`: Two values each, indicating the center and scale of the person in the image.
- `body_pose`: Array of 72 values representing body pose information.
- `has_body_pose`: Boolean (1 or 0) indicating the availability of body pose information.
- `has_betas`: Boolean (1 or 0) indicating the availability of betas (body shape parameters).
- `betas`: Array of 10 values representing body shape parameters.
- `personid`: Identifier (e.g., `0`) uniquely identifying each person in the dataset.

Example Python snippet for creating `.data.pyd` files:
```python
pyd_data = [{
    'keypoints_2d': np.array(keypoints_2d[i], dtype=np.float32),
    'keypoints_3d': np.array(keypoints_3d[i], dtype=np.float32),
    'center': np.array(center[i], dtype=np.float32),
    'scale': np.array(scale[i], dtype=np.float32),
    'body_pose': np.array(body_pose[i].flatten(), dtype=np.float32),
    'betas': np.array(betas[i], dtype=np.float32),
    'has_body_pose': bool(has_body_pose[i]),
    'has_betas': bool(has_betas[i]),
    'personid': 0
}]

# Serialize the data using pickle
with open(new_pyd_path, 'wb') as pyd_file:
    pickle.dump(pyd_data, pyd_file)
```

### Image Formats

Images can be in various formats such as JPEG (`.jpg`), PNG (`.png`), etc.
Ensure all image names and .data.pyd  bases names are same so we can create .tar files. 


#### Data Types

Ensure all data is stored with `numpy.float32` type where applicable to maintain compatibility with PyTorch.

### 3. Data Sorting

When creating the `.tar` archive, ensure data is sorted using the `--sort=name` option to maintain consistent ordering across shards.

Example command for creating `.tar` archive:
```bash
tar --sort=name -cf data.tar /path/to/source/folder
```

For additional guidance on sorting TAR files, refer to [YouTube tutorials](https://www.youtube.com/watch?v=v_PacO-3OGQ) and practical examples available on GitHub.

## Example

![Example Image](https://github.com/user-attachments/assets/bcb8fedb-98f9-4f0f-ae20-6db00d508ef1)

This image illustrates the structure and process of organizing data into the WebDataset format.

## Resources

- WebDataset GitHub Repository: [webdataset/webdataset](https://github.com/webdataset/webdataset)
- YouTube Tutorial on Sorting TAR Files: [Sorting TAR Files Tutorial](https://www.youtube.com/watch?v=v_PacO-3OGQ)

By following these steps and guidelines, you can effectively prepare and utilize datasets in WebDataset format for your PyTorch applications.
