import os
import numpy as np
import rasterio
import laspy
import geopandas as gpd
from tqdm import tqdm

def load_preprocessed_data(file_path):
    data = np.load(file_path)
    tile_image = data['tile_image']  # Shape: (bands, height, width)
    label_array = data['label_array']  # Shape: (classes, height, width)
    superpixel_mask = data['superpixel_mask']  # Shape: (height, width)
    nodata_mask = data['nodata_mask']  # Shape: (height, width)
    return tile_image, label_array, superpixel_mask, nodata_mask

def load_point_cloud(laz_file_path):
    # Read point cloud data from .laz file
    point_cloud = laspy.read(laz_file_path)
    # Extract x, y, z coordinates
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    classifications = point_cloud.classification
    
    # Filter out points with classification 0 (if necessary)
    valid_indices = classifications != 0
    valid_points = points[valid_indices]
    
    return valid_points

def get_polygon_labels(polygon_gdf):
    # Create a dictionary mapping POLYID to specs_perc
    polyid_to_label = {}
    for idx, row in polygon_gdf.iterrows():
        polyid = row['POLYID']
        specs_perc = row['perc_specs']  # Adjust the field name as needed
        polyid_to_label[polyid] = specs_perc
    return polyid_to_label

def save_combined_data(output_file_path, tile_image, label_array, superpixel_mask, nodata_mask, polyid_to_point_cloud, polyid_to_label):
    # Save all data into a compressed .npz file
    np.savez_compressed(
        output_file_path,
        tile_image=tile_image,
        label_array=label_array,
        superpixel_mask=superpixel_mask,
        nodata_mask=nodata_mask,
        polyid_to_point_cloud=polyid_to_point_cloud,
        polyid_to_label=polyid_to_label
    )

def read_split_file(split_file_path):
    with open(split_file_path, 'r') as f:
        tile_names = [line.strip() for line in f]
    return tile_names

def generate_combined_data_for_split(
    tile_npz_dir,
    point_cloud_dir,
    polygon_file_path,
    output_dir,
    split_file_path
):
    # Read the split file
    tile_names = read_split_file(split_file_path)
    
    # Load the polygon file with specs_perc per polygon
    polygon_gdf = gpd.read_file(polygon_file_path)
    polygon_gdf['POLYID'] = polygon_gdf['POLYID'].astype(int)
    polyid_to_label = get_polygon_labels(polygon_gdf)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for tile_name in tqdm(tile_names, desc='Processing Tiles'):
        tile_file = f"{tile_name}.npz"
        tile_file_path = os.path.join(tile_npz_dir, tile_file)
        
        if not os.path.exists(tile_file_path):
            print(f"Tile file {tile_file} not found.")
            continue
        
        # Load preprocessed tile data
        tile_image, label_array, superpixel_mask, nodata_mask = load_preprocessed_data(tile_file_path)
        
        # Get unique POLYIDs in the superpixel mask
        unique_polyids = np.unique(superpixel_mask)
        unique_polyids = unique_polyids[unique_polyids != 0]  # Exclude background or no-data value (assuming 0)
        
        # Initialize dictionaries to store point clouds and labels
        polyid_to_point_cloud = {}
        polyid_to_label_tile = {}
        
        for polyid in unique_polyids:
            # Load point cloud data corresponding to the POLYID
            laz_file_path = os.path.join(point_cloud_dir, f"{polyid}.laz")
            if os.path.exists(laz_file_path):
                point_cloud = load_point_cloud(laz_file_path)
                if point_cloud.size == 0:
                    print(f"No valid points in point cloud for POLYID {polyid}")
                    continue
                polyid_to_point_cloud[polyid] = point_cloud
            else:
                print(f"Point cloud file for POLYID {polyid} not found.")
                continue  # Skip if point cloud file doesn't exist
            
            # Get the label (specs_perc) for the POLYID
            if polyid in polyid_to_label:
                polyid_to_label_tile[polyid] = polyid_to_label[polyid]
            else:
                print(f"Label for POLYID {polyid} not found in polygon file.")
                continue  # Skip if label not found
        
        # Save the combined data
        output_file_path = os.path.join(output_dir, f"{tile_name}_combined.npz")
        save_combined_data(
            output_file_path,
            tile_image,
            label_array,
            superpixel_mask,
            nodata_mask,
            polyid_to_point_cloud,
            polyid_to_label_tile
        )
        print(f"Saved combined data for tile {tile_name} to {output_file_path}")

if __name__ == "__main__":
    # Define paths
    tile_npz_dir = 'path/to/tile_npz_files'  # Directory containing per-tile .npz files
    point_cloud_dir = 'path/to/point_cloud_files'  # Directory containing .laz files per POLYID
    polygon_file_path = 'path/to/polygon_file.gpkg'  # Path to polygon file with specs_perc per polygon
    output_dir = 'path/to/output_combined_data'  # Directory to save combined data files
    split_file_path = 'path/to/split_file.txt'  # Path to the split file (train/test/val)
    
    # Generate combined data for the specified split
    generate_combined_data_for_split(
        tile_npz_dir,
        point_cloud_dir,
        polygon_file_path,
        output_dir,
        split_file_path
    )