# M3F-Net: Fusion of SPL Point Cloud and Sentinel-2 Imagery for Tree Species Composition Estimation  

This repository provides tools and workflows for fusing Single Photon LiDAR (SPL) point clouds and Sentinel-2 satellite imagery to enhance the accuracy of tree species composition estimation. By integrating high-resolution 3D point cloud data with multispectral imagery, this project aims to push the boundaries of precision forestry.  

## Repository Structure  

```
data/
├── preprocessed_tiles/
│   ├── tile_1.npz
│   ├── tile_2.npz
│   └── ...
notebooks/  
├── fri_processing.ipynb             # fri data preprocessing 
├── s2_processing.ipynb              # Sentinel-2 preprocessing
├── superpixel_processing.ipynb      # superpixel data preprocessing  
├── fusion_training.ipynb            # Training and evaluation of fusion models  
dataset/
data_utils/
models/
scripts/
utils/
README.md
```

## Installation  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/yuwei-cao-git/M3F-Net.git  
   cd M3F-Net 
   ```  
2. Install dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  

## Dependencies  
- Python 3.8+  
- Pytorch
- Lightning
- RayTune  
- Open3D  
- Geopandas  
- Rasterio  
- Scikit-learn  
- GDAL  

## Example Use Case  
Run the `fri_processing.ipynb` and `s2_processing.ipynb` notebooks to preprocess fri, SPL, Sentinel-2, and superpixel dataset. Use `fusion_training.ipynb` to train a multi-modal fusion model and estimate tree species composition.  

## Acknowledgments  
We acknowledge the open code provided by [PointNext](https://github.com/kentechx/pointnext/blob/main/pointnext/pointnext.py) and [Unet](https://github.com/jaxony/unet-pytorch).  

## License  
This project is licensed under the [MIT License](LICENSE).  

---  

Let me know if there’s anything you’d like to add or modify!
