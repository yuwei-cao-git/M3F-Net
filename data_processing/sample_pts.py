import laspy
import numpy as np
from shapely import Point
import os
from pathlib import Path
import geopandas as gpd
import numpy as np
from resample_pts import farthest_point_sampling
import argparse

def resample_points_within_polygon(pts, max_pts, min_x, max_x, min_y, max_y, min_z, max_z):
    # Number of points to sample
    num_points = max_pts

    # Randomly sample x, y, and z within the specified bounds
    if pts.shape[0] == 0:
        x = np.random.uniform(min_x, max_x, num_points)
        y = np.random.uniform(min_y, max_y, num_points)
        z = np.random.uniform(min_z, max_z, num_points)
        # Combine into an array of (x, y, z) points
        pts = np.column_stack((x, y, z))
        classification = np.zeros(num_points, dtype=np.uint8)
    elif pts.shape[0] >= max_pts:
        use_idx = farthest_point_sampling(pts, num_points)
        pts = pts[use_idx, :]
        classification = np.ones(num_points, dtype=np.uint8)
    else:
        use_idx = np.random.choice(pts.shape[0], num_points, replace=True)
        pts = pts[use_idx, :]
        classification = np.full(num_points, 2, dtype=np.uint8)
    return pts, classification  

# Function to process and sample points within a polygon
def sample_points_within_polygon(las_file_path, polygon, max_pts, output_folder, polyid, get_attributes=False):
    extracted_points = []
    # Read the LAS file
    inFile = laspy.read(las_file_path)
    
    # Get bounds of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    # Get coordinates (XYZ)
    points = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()

    # Apply the height filter to keep points with z > 2 meters
    height_filtered_points = points[points[:, 2] > 2]

    # Filter points within the bounds
    mask = (height_filtered_points[:, 0] >= minx) & (height_filtered_points[:, 0] <= maxx) & (height_filtered_points[:, 1] >= miny) & (height_filtered_points[:, 1] <= maxy)
    candidate_points = height_filtered_points[mask]

    # Further filter points within the polygon
    for point in candidate_points:
        if polygon.contains(Point(point[0], point[1])):
            extracted_points.append(point)
    extracted_points = np.array(extracted_points)
    
    output_las_file_path = os.path.join(output_folder, f"{polyid}.laz")
        
    # Save the merged points to a new LAS file
    output_las = laspy.create(point_format=inFile.header.point_format, file_version=inFile.header.version)
    
    extracted_points, classification = resample_points_within_polygon(extracted_points, max_pts, minx, maxx, miny, maxy, 2, 3)
    
    output_las.x = extracted_points[:, 0]
    output_las.y = extracted_points[:, 1]
    output_las.z = extracted_points[:, 2]
    
    # Copy over other point attributes if needed (intensity, classification, etc.)
    if get_attributes:
        output_las.classification = classification

    output_las.write(output_las_file_path)
    print(f"Saved sampled points for {polyid} to {output_las_file_path}")
        
def sample_pts(polygons_file_path, las_files_directory, output_folder):
    # Directory containing the LAS files
    las_files_directory = Path(las_files_directory)
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    # Load the polygon GeoDataFrame #32453, 52668, 63838, 64050-51
    gdf_polygons = gpd.read_file(Path(polygons_file_path))[166815:200000]
    # Iterate through each polygon and process the corresponding LAS file
    for _, polygon_row in gdf_polygons.iterrows():
        poly_id = polygon_row['POLYID']
        tilename = polygon_row['Tilename']
        polygon = polygon_row.geometry
        
        # Find the corresponding LAS file
        las_file_path = os.path.join(las_files_directory, f"{tilename}.laz")
        if os.path.exists(las_file_path):
            sample_points_within_polygon(las_file_path, polygon, 7168, output_folder, poly_id, get_attributes=True)
        else:
            print(f"LAS file for {tilename} not found.")
            
            
if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Process some input files for tree species estimation.")
    
    # Add arguments for the input parameters
    parser.add_argument('--polygons_file_path', type=str, required=True,
                        help="Path to the polygons file (e.g., a .gpkg file).")
    parser.add_argument('--las_files_directory', type=str, required=True,
                        help="Directory containing the raw .las files.")
    parser.add_argument('--output_folder', type=str, required=True,
                        help="Directory where the output will be saved.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    sample_pts(args.polygons_file_path, args.las_files_directory, args.output_folder)