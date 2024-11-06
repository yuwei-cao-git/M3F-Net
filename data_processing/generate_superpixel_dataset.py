import os
import numpy as np
import rasterio
import laspy
import geopandas as gpd
from tqdm import tqdm


def load_preprocessed_data(file_path):
    data = np.load(file_path, allow_pickle=True)
    tile_images = data["tile_images"]  # List of images from all seasons
    label_array = data["label_array"]  # Shape: (num_classes, height, width)
    superpixel_mask = data["superpixel_mask"]  # Shape: (height, width)
    nodata_mask = data["nodata_mask"]  # Shape: (height, width)
    return tile_images, label_array, superpixel_mask, nodata_mask


def load_point_cloud(laz_file_path):
    # Read point cloud data from .laz file
    point_cloud = laspy.read(laz_file_path)
    # Extract x, y, z coordinates
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
    classifications = point_cloud.classification

    # Filter out points with classification 0 (invalid points)
    valid_indices = classifications != 0
    valid_points = points[valid_indices]

    return valid_points


def get_polygon_labels(polygon_gdf):
    # Create a dictionary mapping POLYID to specs_perc
    polyid_to_label = {}
    for idx, row in polygon_gdf.iterrows():
        polyid = row["POLYID"]
        specs_perc = row["perc_specs"]  # Adjust the field name as needed
        specs_perc = specs_perc.replace("[", "")
        specs_perc = specs_perc.replace("]", "")
        specs_perc = specs_perc.split(",")
        polyid_to_label[polyid] = [
            float(i) for i in specs_perc
        ]  # convert items in label to float
        # polyid_to_label[polyid] = np.array(specs_perc, dtype=float)
    return polyid_to_label


def save_superpixel_data(
    polyid,
    superpixel_images,
    point_cloud,
    superpixel_label,
    per_pixel_labels,
    output_dir,
):
    output_file_path = os.path.join(output_dir, f"{polyid}.npz")
    np.savez_compressed(
        output_file_path,
        superpixel_images=superpixel_images,  # List of images from all seasons (each padded to 128x128)
        point_cloud=point_cloud,  # Shape: (7168, 3)
        label=superpixel_label,  # Shape: (9,)
        per_pixel_labels=per_pixel_labels,  # Shape: (num_classes, 128, 128)
    )
    print(f"Superpixel data saved to {output_file_path}")


def read_split_file(split_file_path):
    with open(split_file_path, "r") as f:
        tile_names = [line.strip() for line in f]
    return tile_names


def generate_combined_data_for_split(
    tile_npz_dir,
    point_cloud_dir,
    polygon_file_path,
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
        unique_polyids = np.unique(superpixel_mask)
        unique_polyids = unique_polyids[
            unique_polyids != 0
        ]  # Exclude background or no-data value (assuming 0)

        for polyid in unique_polyids:
            # Create a mask for the current superpixel
            superpixel_mask_binary = superpixel_mask == polyid

            # Extract image patches or masked images for each season and pad to (128, 128)
            padded_superpixel_images = []
            for season_image in tile_images:
                # Create a blank padded image of shape (num_channels, 128, 128)
                num_channels = season_image.shape[0]
                padded_image = np.zeros(
                    (num_channels, 128, 128), dtype=season_image.dtype
                )

                # Assign the superpixel values to their respective locations in the padded image
                for channel_idx in range(num_channels):
                    padded_image[channel_idx][superpixel_mask_binary] = season_image[
                        channel_idx
                    ][superpixel_mask_binary]

                # Add the padded image to the list of superpixel images
                padded_superpixel_images.append(padded_image)

            # Load point cloud data corresponding to the POLYID
            laz_file_path = os.path.join(point_cloud_dir, f"{polyid}.laz")
            if os.path.exists(laz_file_path):
                point_cloud = load_point_cloud(laz_file_path)
                if point_cloud.size == 0:
                    print(f"No valid points in point cloud for POLYID {polyid}")
                    continue
            else:
                print(f"Point cloud file for POLYID {polyid} not found.")
                continue  # Skip if point cloud file doesn't exist

            # Get the label (specs_perc) for the POLYID
            if polyid in polyid_to_label:
                superpixel_label = polyid_to_label[polyid]
            else:
                print(f"Label for POLYID {polyid} not found in polygon file.")
                continue  # Skip if label not found

            # Create a blank padded per-pixel label array of shape (num_classes, 128, 128)
            num_classes = label_array.shape[0]
            padded_per_pixel_labels = np.zeros(
                (num_classes, 128, 128), dtype=label_array.dtype
            )

            # Assign the superpixel values to their respective locations in the padded per-pixel label array
            for class_idx in range(num_classes):
                padded_per_pixel_labels[class_idx][superpixel_mask_binary] = (
                    label_array[class_idx][superpixel_mask_binary]
                )

            # Save the superpixel data
            save_superpixel_data(
                polyid,
                padded_superpixel_images,
                point_cloud,
                superpixel_label,
                padded_per_pixel_labels,
                output_superpixel_dir,
            )


if __name__ == "__main__":
    resolutions = ["20m"]
    splits = ["train", "test", "val"]
    for resolution in resolutions:
        tile_npz_dir = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed/{resolution}/rmf_s2_compressed"
        for split in splits:
            point_cloud_dir = "/mnt/g/rmf/m3f_spl/superpxiel_plots/7168"  # Directory containing .laz files per POLYID
            polygon_file_path = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/rmf_plots/fusion/superpixel_plots_{resolution}_Tilename.gpkg"  # Path to polygon file with specs_perc per polygon
            output_superpixel_dir = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed/{resolution}/fusion/{split}/superpixel"

            split_file_path = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed/{resolution}/dataset/{split}_tiles.txt"  # Path to the split file (train/test/val)

            # Generate combined data for the specified split
            generate_combined_data_for_split(
                tile_npz_dir,
                point_cloud_dir,
                polygon_file_path,
                output_superpixel_dir,
                split_file_path,
            )
