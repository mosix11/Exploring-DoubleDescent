import torch
from torch.utils.data import Dataset, DataLoader, random_split
import math
import matplotlib.pyplot as plt # Optional for visualization

class MoGSyntheticDataset(Dataset):
    """
    A PyTorch Dataset class to generate synthetic classification data.

    Generates data based on multimodal Gaussian mixtures, similar to
    sklearn's make_classification, upon initialization. Provides access
    to the generated features and labels. Use with torch.utils.data.random_split
    to create training, validation, and test sets.
    """
    def __init__(
        self,
        n_samples=1000,
        n_features=20,
        n_classes=2,
        n_clusters_per_class=1,
        class_sep=1.0,
        cluster_std=1.0,
        label_noise=0.0,
        shuffle_in_generation=True, # Shuffle points during generation itself
        random_state=None,
        device=None,
        dtype=torch.float32
    ):
        """
        Args:
            n_samples (int): Total number of samples to generate for the entire dataset.
            n_features (int): Number of features for each sample.
            n_classes (int): Number of classes.
            n_clusters_per_class (int): Number of Gaussian clusters per class.
            class_sep (float): Factor scaling cluster center separation.
            cluster_std (float or list/tuple): Standard deviation(s) of the clusters.
            label_noise (float): Fraction of labels to randomly flip (0.0 to 1.0).
            shuffle_in_generation (bool): Shuffle samples internally after generation.
                                         Note: random_split performs further shuffling.
            random_state (int or torch.Generator): Seed or generator for reproducibility.
            device (torch.device or str): Device for tensor creation ('cpu', 'cuda', etc.).
            dtype (torch.dtype): Data type for the feature tensor X.
        """
        super().__init__()
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.n_clusters_per_class = n_clusters_per_class
        self.class_sep = class_sep
        self.cluster_std = cluster_std
        self.label_noise = label_noise
        self.shuffle_in_generation = shuffle_in_generation
        self.device = device
        self.dtype = dtype

        if random_state is None:
            self.generator = None
        elif isinstance(random_state, int):
            self.generator = torch.Generator().manual_seed(random_state)
        elif isinstance(random_state, torch.Generator):
            self.generator = random_state
        else:
            raise ValueError("random_state must be None, int, or torch.Generator")

        # Generate the data internally
        self._generate_data()

    def _generate_data(self):
        """Internal method to generate features (X) and labels (y)."""
        num_total_clusters = self.n_classes * self.n_clusters_per_class
        
        # Avoid division by zero if n_samples is too small or clusters is zero
        if num_total_clusters == 0:
             raise ValueError("n_classes and n_clusters_per_class must be positive.")
             
        samples_per_cluster = self.n_samples // num_total_clusters
        n_samples_generated = samples_per_cluster * num_total_clusters

        if n_samples_generated == 0:
             raise ValueError(f"n_samples ({self.n_samples}) too small for the number of clusters ({num_total_clusters}). Increase n_samples.")

        X_list = []
        y_list = []

        # Generate cluster centers
        cluster_means = torch.randn(
            num_total_clusters, self.n_features,
            generator=self.generator, device=self.device, dtype=self.dtype
        ) * (self.class_sep * 10) # Heuristic scaling for separation

        # Prepare cluster standard deviations
        if isinstance(self.cluster_std, (float, int)):
            cluster_stds = torch.full((num_total_clusters,), float(self.cluster_std), device=self.device, dtype=self.dtype)
        elif isinstance(self.cluster_std, (list, tuple)):
            if len(self.cluster_std) != num_total_clusters:
                raise ValueError("Length of cluster_std must match n_classes * n_clusters_per_class")
            cluster_stds = torch.tensor(self.cluster_std, device=self.device, dtype=self.dtype)
        else:
            raise TypeError("cluster_std must be float, list, or tuple")

        # Sample points from each cluster
        for i in range(num_total_clusters):
            mean = cluster_means[i]
            std = cluster_stds[i]
            n_samples_cluster = samples_per_cluster
            if i == num_total_clusters - 1: # Add remainder samples to the last cluster
                n_samples_cluster += self.n_samples - n_samples_generated

            cluster_X = mean + std * torch.randn(
                n_samples_cluster, self.n_features,
                generator=self.generator, device=self.device, dtype=self.dtype
            )
            X_list.append(cluster_X)

            class_label = i // self.n_clusters_per_class
            cluster_y = torch.full((n_samples_cluster,), class_label, device=self.device, dtype=torch.int64)
            y_list.append(cluster_y)

        # Concatenate all generated data
        self.features = torch.cat(X_list, dim=0)
        self.labels = torch.cat(y_list, dim=0)

        # Apply label noise
        if self.label_noise > 0.0:
            self._apply_label_noise()

        # Shuffle internally if requested
        if self.shuffle_in_generation:
            indices = torch.randperm(self.n_samples, generator=self.generator, device=self.device)
            self.features = self.features[indices]
            self.labels = self.labels[indices]

    def _apply_label_noise(self):
        """Internal method to apply label noise."""
        n_flip = int(self.label_noise * self.n_samples)
        if n_flip > 0:
            # Ensure we don't try to flip more samples than available
            n_flip = min(n_flip, self.n_samples) 
            
            flip_indices = torch.randperm(self.n_samples, generator=self.generator, device=self.device)[:n_flip]
            original_labels = self.labels[flip_indices]

            new_labels = torch.randint(0, self.n_classes, (n_flip,), generator=self.generator, device=self.device, dtype=torch.int64)
            needs_resample = (new_labels == original_labels)
            
            # Keep resampling for indices where the new random label matched the old one
            max_attempts = 10 # Prevent infinite loop in edge cases (e.g., n_classes=1)
            attempts = 0
            while torch.any(needs_resample) and attempts < max_attempts :
                 num_resample = torch.sum(needs_resample).item()
                 current_indices_to_resample = flip_indices[needs_resample] # Get actual indices
                 
                 # Generate new labels *only* for those needing resampling
                 resampled_labels = torch.randint(0, self.n_classes, (num_resample,), generator=self.generator, device=self.device, dtype=torch.int64)
                 
                 # Update the new_labels tensor only at the positions that needed resampling
                 new_labels[needs_resample] = resampled_labels 
                 
                 # Check again which ones still need resampling
                 original_labels_subset = self.labels[current_indices_to_resample] # Get original labels for the subset
                 needs_resample = (resampled_labels == original_labels_subset) # Check the resampled subset

                 # **Important:** Update the global `needs_resample` mask for the *next* potential iteration
                 # Create a full-size mask, default to False
                 full_needs_resample_mask = torch.zeros_like(flip_indices, dtype=torch.bool) 
                 # Set True only at the indices within `flip_indices` that still need resampling
                 full_needs_resample_mask[needs_resample] = True
                 needs_resample = full_needs_resample_mask # Assign back for loop condition
                 
                 attempts += 1
                 
            if attempts == max_attempts and torch.any(needs_resample):
                print(f"Warning: Label noise generation could not ensure all flipped labels are different after {max_attempts} attempts.")

            self.labels[flip_indices] = new_labels


    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """Returns the feature and label for a given index."""
        if not isinstance(idx, int):
             # Handle slicing or advanced indexing if necessary, or raise error
             # For simplicity, let's assume basic integer indexing for now
             raise TypeError("Index must be an integer")
             
        if not (0 <= idx < self.n_samples):
             raise IndexError(f"Index {idx} out of bounds for dataset with length {self.n_samples}")

        return self.features[idx], self.labels[idx]

# --- Example Usage ---

# 1. Define Generation Parameters
data_params = {
    'n_samples': 2000,
    'n_features': 10,
    'n_classes': 4,
    'n_clusters_per_class': 2,
    'class_sep': 4.0,
    'cluster_std': 1.5,
    'label_noise': 0.05, # 5% label noise
    'shuffle_in_generation': True,
    'random_state': 123, # Use a seed for reproducible generation AND splitting
    'device': 'cpu' # Use 'cuda' if available and desired
}

# 2. Instantiate the Full Dataset
# Use a single generator for both dataset creation and splitting for full reproducibility
master_generator = torch.Generator(device=data_params['device']).manual_seed(data_params['random_state'])

full_dataset = MoGSyntheticDataset(
    n_samples=data_params['n_samples'],
    n_features=data_params['n_features'],
    n_classes=data_params['n_classes'],
    n_clusters_per_class=data_params['n_clusters_per_class'],
    class_sep=data_params['class_sep'],
    cluster_std=data_params['cluster_std'],
    label_noise=data_params['label_noise'],
    shuffle_in_generation=data_params['shuffle_in_generation'],
    random_state=master_generator, # Pass the generator
    device=data_params['device']
)

print(f"Total dataset size: {len(full_dataset)}")
# Accessing a single item:
# feature_example, label_example = full_dataset[0]
# print("Example item 0:", feature_example.shape, label_example.item())


# 3. Define Split Sizes (e.g., 70% train, 15% validation, 15% test)
total_size = len(full_dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size # Ensure it sums up

print(f"Splitting into: Train={train_size}, Val={val_size}, Test={test_size}")

# 4. Perform the Split using random_split
# Pass the same generator to random_split for reproducible splits
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=master_generator # Reuse the generator
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Accessing an item from a subset (note: indices are relative to the subset)
# train_feature_0, train_label_0 = train_dataset[0]
# print("Example item 0 from train split:", train_feature_0.shape, train_label_0.item())


# 5. Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # No need to shuffle val/test
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("\nCreated DataLoaders.")

# Example of iterating through a DataLoader
print("\nExample batch from train_loader:")
for batch_idx, (features, labels) in enumerate(train_loader):
    print(f"  Batch {batch_idx + 1}:")
    print(f"    Features shape: {features.shape}") # [batch_size, n_features]
    print(f"    Labels shape: {labels.shape}")     # [batch_size]
    # print(f"    Labels: {labels}")
    break # Just show the first batch

# --- Optional: Visualize one of the splits if n_features == 2 ---
if data_params['n_features'] == 2:
    # Get all data from the validation set for plotting
    val_features_list = []
    val_labels_list = []
    for features, labels in DataLoader(val_dataset, batch_size=len(val_dataset)): # Load all at once
        val_features_list.append(features)
        val_labels_list.append(labels)
    val_features = torch.cat(val_features_list).cpu().numpy()
    val_labels = torch.cat(val_labels_list).cpu().numpy()

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(val_features[:, 0], val_features[:, 1], c=val_labels, cmap='viridis', s=15, alpha=0.7)
    plt.title(f'Validation Set Data ({len(val_dataset)} samples)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    try:
        handles, _ = scatter.legend_elements()
        plt.legend(handles, [f'Class {i}' for i in range(data_params['n_classes'])], title="Classes")
    except Exception:
        print("Could not generate legend.")
    plt.grid(True)
    plt.show()