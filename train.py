from torch.utils.data import DataLoader

def load_tile_names(file_path):
    """
    Load tile names from a .txt file.

    Args:
        file_path (str): Path to the .txt file.

    Returns:
        tile_names (list): List of tile names.
    """
    with open(file_path, 'r') as f:
        tile_names = f.read().splitlines()
    return tile_names

# Directories
tiles_dir = "s2/tiles_128"  # Directory containing the 12-band Sentinel imagery
labels_dir = "labels/tiles_128"  # Directory containing the 9-band species proportions

# Load tile names for each split
train_tile_names = load_tile_names("train.txt")
val_tile_names = load_tile_names("validation.txt")
test_tile_names = load_tile_names("test.txt")

# Create datasets
train_dataset = TreeSpeciesDataset(tiles_dir, labels_dir, train_tile_names)
val_dataset = TreeSpeciesDataset(tiles_dir, labels_dir, val_tile_names)
test_dataset = TreeSpeciesDataset(tiles_dir, labels_dir, test_tile_names)

# Create DataLoaders
batch_size = 4  # Adjust based on your memory capacity
num_workers = 4  # Adjust based on your system

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

import torch.optim as optim

# Initialize the model, loss function, and optimizer
model = UNet(in_channels=12, out_channels=9)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Training parameters
num_epochs = 10  # Adjust as needed

# Training loop
# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets, mask in train_loader:
        inputs = inputs.to(device)  # Shape: (batch_size, 12, H, W)
        targets = targets.to(device)  # Shape: (batch_size, 9, H, W)
        mask = mask.to(device)  # Shape: (batch_size, H, W)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)  # Shape: (batch_size, 9, H, W)

        # Apply the mask to ignore NoData pixels in the loss calculation
        valid_outputs = outputs[:, :, ~mask]
        valid_targets = targets[:, :, ~mask]

        # Compute loss only for valid pixels
        loss = criterion(valid_outputs, valid_targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_loss:.4f}")

    # Validation step
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets, mask in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            outputs = model(inputs)

            # Apply the mask to ignore NoData pixels
            valid_outputs = outputs[:, :, ~mask]
            valid_targets = targets[:, :, ~mask]

            loss = criterion(valid_outputs, valid_targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

# Save the trained model
model_save_path = "tree_species_model.pth"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")