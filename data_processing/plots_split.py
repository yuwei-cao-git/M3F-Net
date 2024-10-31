import geopandas as gpd
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import math

def load_tile_splits(folder_path):
    """Load train/val/test tile splits from a folder containing the text files."""
    train_file = os.path.join(folder_path, 'train_tiles.txt')
    val_file = os.path.join(folder_path, 'val_tiles.txt')
    test_file = os.path.join(folder_path, 'test_tiles.txt')

    with open(train_file, 'r') as f:
        train_tiles = f.read().splitlines()
    with open(val_file, 'r') as f:
        val_tiles = f.read().splitlines()
    with open(test_file, 'r') as f:
        test_tiles = f.read().splitlines()

    return train_tiles, val_tiles, test_tiles

def get_tile_coordinates(tile_name):
    """Extract the i, j coordinates of the tile from its name (assuming 'tile_i_j' format)."""
    _, i, j = tile_name.split('_')
    return int(i), int(j)

def assign_tile_to_plots(plots_gpkg, train_tiles, val_tiles, test_tiles, raster_width=5807, tile_step=128):
    """Assign each plot to the corresponding tile based on its ID without recalculating geometry."""
    # Load the plots from the GeoPackage
    plots = gpd.read_file(plots_gpkg)

    # Add new columns for tile_name, tile_i, tile_j
    plots['tile_name'] = None
    plots['tile_i'] = None
    plots['tile_j'] = None

    # Create empty GeoDataFrames for train, val, and test sets
    train_plots = gpd.GeoDataFrame(columns=plots.columns, crs=plots.crs)
    val_plots = gpd.GeoDataFrame(columns=plots.columns, crs=plots.crs)
    test_plots = gpd.GeoDataFrame(columns=plots.columns, crs=plots.crs)

    # Assign tile name, tile_i, tile_j based on plot ID and raster dimensions
    for index, plot in plots.iterrows():
        # Plot ID is the pixel's ID, which can be converted to row and col
        pixel_id = plot['id']
        row = pixel_id // raster_width  # Get the row index in the original raster
        col = pixel_id % raster_width   # Get the column index in the original raster

        # Determine which tile the plot belongs to
        tile_i = row // tile_step
        tile_j = col // tile_step
        tile_name = f"tile_{tile_i}_{tile_j}"

        # Assign tile_name, tile_i, tile_j to the plot's attributes
        plots.at[index, 'tile_name'] = tile_name
        plots.at[index, 'tile_i'] = tile_i
        plots.at[index, 'tile_j'] = tile_j

        # Assign the plot to the correct split based on the tile
        if tile_name in train_tiles:
            train_plots = train_plots.append(plots.loc[index], ignore_index=True)
        elif tile_name in val_tiles:
            val_plots = val_plots.append(plots.loc[index], ignore_index=True)
        elif tile_name in test_tiles:
            test_plots = test_plots.append(plots.loc[index], ignore_index=True)

    return train_plots, val_plots, test_plots

def save_splits(train_plots, val_plots, test_plots):
    """Save the split GeoDataFrames to separate GeoPackage files."""
    train_plots.to_file('plot_train.gpkg', driver="GPKG")
    val_plots.to_file('plot_val.gpkg', driver="GPKG")
    test_plots.to_file('plot_test.gpkg', driver="GPKG")

def plot_tile_with_plots(train_plots, tile_name):
    """Plot a tile and corresponding plots from the train split."""
    # Filter plots based on tile_name
    tile_plots = train_plots[train_plots['tile_name'] == tile_name]

    # Plot the tile area
    plt.figure(figsize=(8, 8))
    tile_plots.plot(ax=plt.gca(), color='red', markersize=2)
    plt.title(f"Tile {tile_name} with Plots from Train Split")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.show()

def plot_pixel_level_cloud(train_plots, tile_name):
    """Plot pixel-level point clouds within a specific tile."""
    # Filter plots based on tile_name
    pixel_plots = train_plots[train_plots['tile_name'] == tile_name]

    # Plot pixel-level point cloud
    plt.figure(figsize=(8, 8))
    plt.scatter(pixel_plots.geometry.x, pixel_plots.geometry.y, s=1, color='blue')
    plt.title(f"Pixel-level Point Cloud for Tile {tile_name}")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.show()

# Main function to execute the splitting process
def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Split plots into train/val/test GeoPackages based on tile splits and visualize.")
    parser.add_argument("--folder_path", type=str, required=True, help="Folder path containing train/val/test files.")
    parser.add_argument("--plots_gpkg", type=str, required=True, help="File path for plots.gpkg.")
    parser.add_argument("--raster_width", type=int, required=True, help="Width of the original raster.")
    parser.add_argument("--tile_name", type=str, required=False, help="Tile name to plot (e.g., 'tile_0_0') for evaluation.", default=None)
    parser.add_argument("--pixel_cloud", action='store_true', help="Flag to plot pixel-level point clouds for the specified tile.")

    args = parser.parse_args()

    # Load tile splits from the user-provided folder
    train_tiles, val_tiles, test_tiles = load_tile_splits(args.folder_path)

    # Assign plots to train/val/test sets based on their tile and add tile_name, tile_i, tile_j
    train_plots, val_plots, test_plots = assign_tile_to_plots(args.plots_gpkg, train_tiles, val_tiles, test_tiles, args.raster_width, args.raster_height)

    # Save the splits to separate GeoPackage files
    save_splits(train_plots, val_plots, test_plots)

    # Plot a tile from the train split for evaluation if a tile name is provided
    if args.tile_name:
        if args.pixel_cloud:
            plot_pixel_level_cloud(train_plots, args.tile_name)
        else:
            plot_tile_with_plots(train_plots, args.tile_name)

if __name__ == "__main__":
    main()
