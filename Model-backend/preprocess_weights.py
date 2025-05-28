import h5py
import numpy as np

# Paths to the original and renamed weights files
ORIGINAL_WEIGHTS_PATH = "model_weights.weights.h5"
RENAMED_WEIGHTS_PATH = "model_weights.weights.h5"  # Overwrite the original file

# Function to inspect HDF5 file structure
def inspect_hdf5_group(group, prefix=""):
    for key in group.keys():
        item = group[key]
        if isinstance(item, h5py.Dataset):
            print(f"{prefix}{key}: Shape: {item.shape}")
        elif isinstance(item, h5py.Group):
            inspect_hdf5_group(item, prefix=f"{prefix}{key}/")

# Step 1: Inspect the original weights file
print("Step 1: Inspecting original weights file...")
try:
    with h5py.File(ORIGINAL_WEIGHTS_PATH, "r") as f:
        print("Weights file top-level keys:", list(f.keys()))
        inspect_hdf5_group(f)
except Exception as e:
    print(f"Error inspecting original weights file: {str(e)}")
    exit(1)

# Step 2: Extract the weights in a separate with block
print("Step 2: Extracting weights...")
try:
    with h5py.File(ORIGINAL_WEIGHTS_PATH, "r") as f:
        print("Confirming file structure before extraction...")
        print("Weights file top-level keys:", list(f.keys()))
        if 'layers' not in f:
            print("Error: 'layers' group not found in weights file.")
            exit(1)
        if 'embedding' not in f['layers']:
            print("Error: 'embedding' group not found in weights file.")
            exit(1)
        if 'vars' not in f['layers']['embedding']:
            print("Error: 'vars' group not found in weights file.")
            exit(1)
        if '0' not in f['layers']['embedding']['vars']:
            print("Error: '0' dataset not found in weights file.")
            exit(1)
        print("Extracting embedding weights...")
        embedding_weights = np.array(f['layers']['embedding']['vars']['0'])
        print(f"Extracted embedding weights shape: {embedding_weights.shape}")
        
        print("Extracting dense layer weights...")
        dense_weights = np.array(f['layers']['dense']['vars']['0'])
        dense_bias = np.array(f['layers']['dense']['vars']['1'])
        print(f"Extracted dense weights shape: {dense_weights.shape}, bias shape: {dense_bias.shape}")
        
        print("Extracting dense_1 layer weights...")
        dense_1_weights = np.array(f['layers']['dense_1']['vars']['0'])
        dense_1_bias = np.array(f['layers']['dense_1']['vars']['1'])
        print(f"Extracted dense_1 weights shape: {dense_1_weights.shape}, bias shape: {dense_1_bias.shape}")
except Exception as e:
    print(f"Error extracting weights: {str(e)}")
    exit(1)

# Step 3: Create a new HDF5 file with the renamed weights
print("Step 3: Creating new weights file with renamed weights...")
try:
    with h5py.File(RENAMED_WEIGHTS_PATH, "w") as f_new:
        # Create new groups with the expected Keras naming
        # Embedding layer
        f_new.create_group('embedding')
        f_new['embedding'].create_group('embeddings')
        f_new['embedding']['embeddings']['embeddings:0'] = embedding_weights
        print("Added embedding weights to 'embedding/embeddings:0'.")
        # Dense layer
        f_new.create_group('dense')
        f_new['dense']['kernel:0'] = dense_weights
        f_new['dense']['bias:0'] = dense_bias
        print("Added dense weights to 'dense/kernel:0' and 'dense/bias:0'.")
        # Dense_1 layer
        f_new.create_group('dense_1')
        f_new['dense_1']['kernel:0'] = dense_1_weights
        f_new['dense_1']['bias:0'] = dense_1_bias
        print("Added dense_1 weights to 'dense_1/kernel:0' and 'dense_1/bias:0'.")
except Exception as e:
    print(f"Error creating new weights file: {str(e)}")
    exit(1)

# Step 4: Inspect the new weights file
print("Step 4: Inspecting new weights file...")
try:
    with h5py.File(RENAMED_WEIGHTS_PATH, "r") as f_new:
        print("Weights file top-level keys:", list(f_new.keys()))
        inspect_hdf5_group(f_new)
except Exception as e:
    print(f"Error inspecting new weights file: {str(e)}")
    exit(1)

print("Weights preprocessing completed.")