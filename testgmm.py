import torch
import math
import matplotlib.pyplot as plt # Optional for visualization

def make_classification_pytorch(
    n_samples=100,
    n_features=20,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=1.0,
    cluster_std=1.0,
    label_noise=0.0,
    shuffle=True,
    random_state=None,
    device=None,
    dtype=torch.float32
):
    """
    Generates synthetic classification data using PyTorch, similar to sklearn's make_classification.

    Args:
        n_samples (int): Total number of samples to generate.
        n_features (int): Number of features for each sample.
        n_classes (int): Number of classes.
        n_clusters_per_class (int): Number of Gaussian clusters contributing to each class.
        class_sep (float): Factor multiplying the distance between cluster centers.
                           Larger values create more separated classes.
        cluster_std (float or list/tuple): Standard deviation of the clusters. If a float,
                                         all clusters have the same std. If a list/tuple,
                                         must match the total number of clusters.
        label_noise (float): Fraction of labels to randomly flip (0.0 to 1.0).
        shuffle (bool): Whether to shuffle the samples and labels.
        random_state (int or torch.Generator): Seed or generator for reproducibility.
        device (torch.device or str): Device to create tensors on ('cpu', 'cuda', etc.).
        dtype (torch.dtype): Data type for the feature tensor X.

    Returns:
        (torch.Tensor, torch.Tensor): Tuple containing:
            - X: Feature tensor shape (n_samples, n_features), dtype=dtype.
            - y: Label tensor shape (n_samples,), dtype=torch.int64.
    """
    if random_state is None:
        generator = None
    elif isinstance(random_state, int):
        generator = torch.Generator(device=device).manual_seed(random_state)
    elif isinstance(random_state, torch.Generator):
        generator = random_state
    else:
        raise ValueError("random_state must be None, int, or torch.Generator")

    num_total_clusters = n_classes * n_clusters_per_class
    samples_per_cluster = n_samples // num_total_clusters
    n_samples_generated = samples_per_cluster * num_total_clusters # Ensure divisibility

    if n_samples_generated == 0:
         raise ValueError(f"n_samples ({n_samples}) too small for the number of clusters ({num_total_clusters}).")


    X_list = []
    y_list = []

    # Generate cluster centers (means) somewhat separated
    # Simple approach: sample from a wider Gaussian and scale by class_sep
    # A more sophisticated approach might place them more deliberately (like sklearn's hypercube)
    cluster_means = torch.randn(
        num_total_clusters, n_features,
        generator=generator, device=device, dtype=dtype
    ) * (class_sep * 10) # Scale initial random means for separation


    if isinstance(cluster_std, (float, int)):
        cluster_stds = torch.full((num_total_clusters,), float(cluster_std), device=device, dtype=dtype)
    elif isinstance(cluster_std, (list, tuple)):
        if len(cluster_std) != num_total_clusters:
            raise ValueError("Length of cluster_std must match n_classes * n_clusters_per_class")
        cluster_stds = torch.tensor(cluster_std, device=device, dtype=dtype)
    else:
        raise TypeError("cluster_std must be float, list, or tuple")


    current_sample_idx = 0
    for i in range(num_total_clusters):
        mean = cluster_means[i]
        std = cluster_stds[i]
        n_samples_cluster = samples_per_cluster

        # Handle potential remainder if n_samples wasn't perfectly divisible
        if i == num_total_clusters - 1:
            n_samples_cluster += n_samples - n_samples_generated # Add remainder to last cluster


        # Sample points for the current cluster
        cluster_X = mean + std * torch.randn(
            n_samples_cluster, n_features,
            generator=generator, device=device, dtype=dtype
        )
        X_list.append(cluster_X)

        # Assign class label (integer division maps clusters to classes)
        class_label = i // n_clusters_per_class
        cluster_y = torch.full((n_samples_cluster,), class_label, device=device, dtype=torch.int64)
        y_list.append(cluster_y)


    X = torch.cat(X_list, dim=0)
    y = torch.cat(y_list, dim=0)

    # Apply label noise if requested
    if label_noise > 0.0:
        n_flip = int(label_noise * n_samples)
        if n_flip > 0:
            flip_indices = torch.randperm(n_samples, generator=generator, device=device)[:n_flip]
            original_labels = y[flip_indices]

            # Generate new random labels different from the original
            # Avoid assigning the same label back
            new_labels = torch.randint(0, n_classes, (n_flip,), generator=generator, device=device, dtype=torch.int64)
            # Find indices where new label is same as old, and resample until different
            needs_resample = (new_labels == original_labels)
            while torch.any(needs_resample):
                 num_resample = torch.sum(needs_resample).item()
                 new_labels[needs_resample] = torch.randint(0, n_classes, (num_resample,), generator=generator, device=device, dtype=torch.int64)
                 needs_resample = (new_labels == original_labels) # Check again

            y[flip_indices] = new_labels


    # Shuffle data if requested
    if shuffle:
        indices = torch.randperm(n_samples, generator=generator, device=device)
        X = X[indices]
        y = y[indices]

    return X, y

# --- Example Usage ---
n_samples = 500
n_features = 2
n_classes = 3
n_clusters = 2 # Clusters per class
noise_fraction = 0.0 # 10% label noise

X, y = make_classification_pytorch(
    n_samples=n_samples,
    n_features=n_features,
    n_classes=n_classes,
    n_clusters_per_class=n_clusters,
    class_sep=3.0,       # Increase separation
    cluster_std=10.0,     # Standard deviation within clusters
    label_noise=noise_fraction,
    random_state=111,     # For reproducibility
    device='cpu'         # Or 'cuda' if available
)

print("Generated data shapes:")
print("X:", X.shape, X.dtype)
print("y:", y.shape, y.dtype)
print("\nExample Labels (first 20):")
print(y[:20])
print(f"\nNumber of unique labels: {torch.unique(y).size(0)}")


# --- Optional: Visualize if n_features == 2 ---
if n_features == 2:
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0].cpu().numpy(), X[:, 1].cpu().numpy(), c=y.cpu().numpy(), cmap='viridis', s=15, alpha=0.7)
    plt.title(f'PyTorch Generated Classification Data ({noise_fraction*100:.0f}% Noise)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    try:
        # Create legend handles manually if needed
        handles, _ = scatter.legend_elements()
        plt.legend(handles, [f'Class {i}' for i in range(n_classes)], title="Classes")
    except Exception: # Handle cases where legend elements might fail (e.g., single class)
        print("Could not generate legend.")
    plt.grid(True)
    plt.show()