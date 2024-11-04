import os
import numpy as np
import rasterio
import laspy
import geopandas as gpd
from tqdm import tqdm


def load_preprocessed_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    tile_images = data["tile_images"]  # List of images from all seasons
    label_array = data["label_array"]
    superpixel_mask = data["superpixel_mask"]
    nodata_mask = data["nodata_mask"]
    return tile_images, label_array, superpixel_mask, nodata_mask


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
        polyid = row["POLYID"]
        specs_perc = row["perc_specs"]  # Adjust the field name as needed
        polyid_to_label[polyid] = specs_perc
    return polyid_to_label


def save_superpixel_data(polyid, superpixel_images, point_cloud, label, output_dir):
    output_file_path = os.path.join(output_dir, f"{polyid}.npz")
    np.savez_compressed(
        output_file_path,
        superpixel_images=superpixel_images,
        point_cloud=point_cloud,
        label=label,
    )
    print(f"Superpixel data saved to {output_file_path}")


def save_combined_data(
    output_file_path,
    tile_image,
    label_array,
    superpixel_mask,
    nodata_mask,
    polyid_to_point_cloud,
    polyid_to_label,
):
    # Save all data into a compressed .npz file
    np.savez_compressed(
        output_file_path,
        tile_image=tile_image,
        label_array=label_array,
        superpixel_mask=superpixel_mask,
        nodata_mask=nodata_mask,
        polyid_to_point_cloud=polyid_to_point_cloud,
        polyid_to_label=polyid_to_label,
    )


def read_split_file(split_file_path):
    with open(split_file_path, "r") as f:
        tile_names = [line.strip() for line in f]
    return tile_names


def generate_combined_data_for_split(
    tile_npz_dir,
    point_cloud_dir,
    polygon_file_path,
    output_dir,
    output_superpixel_dir,
    split_file_path,
):
    # Read the split file
    tile_names = read_split_file(split_file_path)

    # Load the polygon file with specs_perc per polygon
    polygon_gdf = gpd.read_file(polygon_file_path)
    polygon_gdf["POLYID"] = polygon_gdf["POLYID"].astype(int)
    polyid_to_label = get_polygon_labels(polygon_gdf)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_superpixel_dir, exist_ok=True)

    for tile_name in tqdm(tile_names, desc="Processing Tiles"):
        tile_file_path = os.path.join(tile_npz_dir, f"{tile_name}_combined.npz")

        # Check that all tile files exist
        if not os.path.exists(tile_file_path):
            print(f"Tile files for {tile_name} not found")
            continue

        # Load preprocessed tile data from all seasons
        tile_images, label_array, superpixel_mask, nodata_mask = load_preprocessed_data(
            tile_file_path
        )

        # Get unique superpixel IDs (excluding background)
        superpixel_ids, counts = np.unique(superpixel_mask, return_counts=True)

        # Create a boolean mask to exclude background (0)
        mask = superpixel_ids != 0  # Exclude background (0)
        # Apply the mask to superpixel_ids and counts
        superpixel_ids = superpixel_ids[mask]
        counts = counts[mask]

        # Create a dictionary of superpixel sizes
        superpixel_sizes = dict(zip(superpixel_ids, counts))

        # Identify small superpixels to remove
        small_superpixels = [
            sp_id for sp_id, size in superpixel_sizes.items() if size < 25
        ]

        # Set small superpixels to background (0)
        for sp_id in small_superpixels:
            superpixel_mask[superpixel_mask == sp_id] = 0

        # Get unique POLYIDs in the superpixel mask
        unique_polyids = np.unique(superpixel_mask)
        unique_polyids = unique_polyids[
            unique_polyids != 0
        ]  # Exclude background or no-data value (assuming 0)

        # Initialize dictionaries to store point clouds and labels
        polyid_to_point_cloud = {}
        polyid_to_label_tile = {}

        for polyid in unique_polyids:
            # Create a mask for the current superpixel
            superpixel_mask_binary = superpixel_mask == polyid

            # Extract image patches or masked images for each season
            superpixel_images = [
                tile_image[:, superpixel_mask_binary] for tile_image in tile_images
            ]

            # Handle nodata pixels if necessary
            superpixel_nodata_mask = nodata_mask[superpixel_mask_binary]

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
                label = polyid_to_label.get(polyid)
            else:
                print(f"Label for POLYID {polyid} not found in polygon file.")
                continue  # Skip if label not found

            save_superpixel_data(
                polyid, superpixel_images, point_cloud, label, output_superpixel_dir
            )

        # Save the combined data
        output_file_path = os.path.join(output_dir, f"{tile_name}.npz")
        save_combined_data(
            output_file_path,
            tile_images,
            label_array,
            superpixel_mask,
            nodata_mask,
            polyid_to_point_cloud,
            polyid_to_label_tile,
        )
        print(f"Saved combined data for tile {tile_name} to {output_file_path}")


if __name__ == "__main__":
    resolutions = ["20m"]
    splits = ["train", "test", "val"]
    for resolution in resolutions:
        tile_npz_dir = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed/{resolution}/rmf_s2_compressed"
        for split in splits:
            point_cloud_dir = "/mnt/g/rmf/m3f_spl/superpxiel_plots/7168"  # Directory containing .laz files per POLYID
            polygon_file_path = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/rmf_plots/fusion/superpixel_plots_{resolution}_Tilename.gpkg"  # Path to polygon file with specs_perc per polygon
            output_dir = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed/{resolution}/fusion/{split}/tile_128"  # Directory to save combined data files
            output_superpixel_dir = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed/{resolution}/fusion/{split}/superpixel"
            split_file_path = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed/{resolution}/dataset/{split}_tiles.txt"  # Path to the split file (train/test/val)
            # Generate combined data for the specified split
            generate_combined_data_for_split(
                tile_npz_dir,
                point_cloud_dir,
                polygon_file_path,
                output_dir,
                output_superpixel_dir,
                split_file_path,
            )
