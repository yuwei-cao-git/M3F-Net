import os
import glob
import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import box
from rasterio.features import rasterize
from tqdm import tqdm
from joblib import Parallel, delayed
import json

def filter_fri_shapefile(fri_shapefile_path, output_shapefile_path, pids_to_keep):
    # Load the FRI shapefile
    fri_gdf = gpd.read_file(fri_shapefile_path)
    fri_gdf['POLYID'] = fri_gdf['POLYID'].astype(int)
    
    # Filter the GeoDataFrame
    filtered_gdf = fri_gdf[fri_gdf['POLYID'].isin(pids_to_keep)]
    
    # Save the filtered shapefile
    filtered_gdf.to_file(output_shapefile_path, driver='ESRI Shapefile')
    
    print(f"Filtered shapefile saved to {output_shapefile_path}")
    
def generate_superpixels(tile_dir, season, resolution, size_threshold):
    
    shapefile = r"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/rmf_fri/fri_rmf_cleaned_9class.gpkg"
    output_dir = f"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed/{resolution}/rmf_s2/{season}/superpixel"  # Update this path
    out_shapefile = r"/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/rmf_fri/superpixel.shp"
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # List to collect all 'pid's used (optional)
    all_pids = set()
    
    # Get list of tile files
    tile_paths = glob.glob(os.path.join(tile_dir, '*.tif'))

    # Load the polygons
    polygons_gdf = gpd.read_file(shapefile)
    print(f"Loaded {len(polygons_gdf)} polygons from {shapefile}")

    # Ensure 'polyid' is of integer type
    if 'POLYID' not in polygons_gdf.columns:
        raise ValueError("Polygons must have a 'polyid' column.")
    polygons_gdf['POLYID'] = polygons_gdf['POLYID'].astype(int)

    # Ensure 'perc_specs' column exists and is properly formatted
    if 'perc_specs' not in polygons_gdf.columns:
        raise ValueError("Polygons must have a 'perc_specs' column containing the 9-class vector labels.")

    # Process each tile
    for tile_path in tqdm(tile_paths, desc='Processing tiles'):
        tile_name = os.path.splitext(os.path.basename(tile_path))[0]
        output_file = os.path.join(output_dir, f"{tile_name}.npz")

        print(f"\nProcessing tile: {tile_name}")

        # Load the tile image
        with rasterio.open(tile_path) as src:
            tile_image = src.read()  # Shape: (bands, height, width)
            tile_transform = src.transform
            tile_crs = src.crs
            tile_bounds = src.bounds
            tile_height, tile_width = src.height, src.width
            nodata_value = src.nodata
            print(f"Nodata value for tile {tile_name}: {nodata_value}")

        # Identify no-data pixels
        if nodata_value is not None:
            nodata_mask = np.any(tile_image == nodata_value, axis=0)  # Shape: (height, width)
        else:
            nodata_mask = np.zeros((tile_height, tile_width), dtype=bool)

        # Reproject polygons to match the tile CRS if necessary
        if polygons_gdf.crs != tile_crs:
            polygons_gdf = polygons_gdf.to_crs(tile_crs)
            print(f"Reprojected polygons to match tile CRS: {tile_crs}")

        # Select polygons that intersect with the tile
        tile_bbox = box(*tile_bounds)
        intersecting_polygons = polygons_gdf[polygons_gdf.intersects(tile_bbox)].copy()

        if intersecting_polygons.empty:
            print(f"No polygons intersect with tile {tile_name}. Excluding this tile.")
            # os.remove(tile_path)
            continue  # Skip this tile

        print(f"Found {len(intersecting_polygons)} intersecting polygons.")

        # Adjust geometries to the tile's coordinate space
        intersecting_polygons['geometry'] = intersecting_polygons.geometry.map(lambda geom: geom.intersection(tile_bbox))

        # Prepare shapes for rasterization using 'POLYID' as the ID
        shapes = zip(intersecting_polygons.geometry, intersecting_polygons['POLYID'])

        # Rasterize polygons onto the tile grid
        superpixel_mask = rasterize(
            shapes=shapes,
            out_shape=(tile_height, tile_width),
            transform=tile_transform,
            fill=0,  # Background value for pixels not covered by any polygon
            all_touched=True,
            dtype='int32'
        )

        # Set superpixel IDs to zero for no-data pixels
        superpixel_mask[nodata_mask] = 0

        # Initialize label array
        num_classes = 9  # Number of classes in the label vector
        label_array = np.zeros((num_classes, tile_height, tile_width), dtype=np.float32)

        # Ensure 'perc_specs' are in the correct format
        intersecting_polygons['perc_specs'] = intersecting_polygons['perc_specs'].apply(lambda x: np.array(json.loads(x), dtype=np.float32))
        polygon_id_to_label = dict(zip(intersecting_polygons['POLYID'], intersecting_polygons['perc_specs']))

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
        small_superpixels = [sp_id for sp_id, size in superpixel_sizes.items() if size < size_threshold]
        
        # Set small superpixels to background (0)
        for sp_id in small_superpixels:
            superpixel_mask[superpixel_mask == sp_id] = 0
        
        # Update the list of superpixel IDs after filtering
        superpixel_ids = np.unique(superpixel_mask)
        superpixel_ids = superpixel_ids[superpixel_ids != 0]  # Exclude background (0)
        
        # Record the 'pid's of intersecting polygons (from superpixel_ids)
        pids_to_record = superpixel_ids.tolist()
        
        all_pids.update(pids_to_record)

        for sp_id in superpixel_ids:
            indices = np.where(superpixel_mask == sp_id)
            label_vector = polygon_id_to_label[sp_id]  # Shape: (9,)
            if label_vector.shape[0] != num_classes:
                raise ValueError(f"Label vector for polyid {sp_id} does not have {num_classes} elements.")
            # Assign the label to the corresponding pixels
            for band in range(num_classes):
                label_array[band, indices[0], indices[1]] = label_vector[band]

        # Handle no-data pixels in labels by setting them to zero
        label_array[:, nodata_mask] = 0.0

        # Handle no-data pixels in imagery data
        tile_image[:, nodata_mask] = 0.0

        # Save preprocessed data
        np.savez_compressed(
            output_file,
            tile_image=tile_image,
            label_array=label_array,
            superpixel_mask=superpixel_mask,
            nodata_mask=nodata_mask
        )
        
        filter_fri_shapefile(shapefile, out_shapefile, all_pids)
        print(f"Total unique pids collected: {len(all_pids)}")
        print(f"Saved preprocessed data to {output_file}")

if __name__ == "__main__":
    seasons = ["spring"]
    resolutions = ["20m"]
    for resolution in resolutions:
        for season in seasons:
            tile_dir = f'/mnt/d/Sync/research/tree_species_estimation/tree_dataset/rmf/processed/{resolution}/rmf_s2/{season}/tiles_128'  # Update this path
            tile_paths = glob.glob(os.path.join(tile_dir, '*.tif'))
            size_threshold = 25 # 10000 m2
            print("in total of " + str(len(tile_paths)) + " tiles")
            # Parallel(n_jobs=-1)(delayed(generate_superpixels)(tile_path) for tile_path in tqdm(tile_paths))
            generate_superpixels(tile_dir, season, resolution, size_threshold)

    '''
    python generate_superpixels.py --tile_dir path/to/tiles/ \
                               --polygon_shapefile path/to/polygons.shp \
                               --output_dir path/to/output/
    '''