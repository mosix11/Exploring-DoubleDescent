import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np # For converting tensors to numpy for plotting

from src.datasets import MoGSyntheticDataset, MoGSynthetic

seed = 10

dataset = MoGSynthetic(
    batch_size=1024,
    num_samples=100000,
    num_features=512,
    num_classes=30,
    clusters_per_class='random',
    base_cluster_std='random',
    covariance_type='full',
    class_sep=2.0,
    intra_class_spread=1.0,
    train_val_test_ratio=[0.7, 0.0, 0.3],
    num_workers=2,
    seed=seed
)

trdl = dataset.get_train_dataloader()
tsdl = dataset.get_test_dataloader()


for batch in trdl:
    # print(len(batch))
    # print(batch[0].shape)
    # print(batch[1].shape)
    print(torch.norm(batch[0][0]))

    
# if __name__ == '__main__':

#     # Parameters for dataset generation
#     N_SAMPLES = 600             # Total number of samples
#     N_FEATURES = 2              # Must be 2 for 2D visualization
#     N_CLASSES = 3               # Number of classes
#     N_CLUSTERS_PER_CLASS = [1, 2, 3]    # Number of clusters within each class (Class 0: 1, Class 1: 2, Class 2: 3)
#     INTRA_CLASS_SEP = 3.0       # Renamed variable for clarity, matches param intra_class_spread
#     CLASS_SEP = 0.5
#     RANDOM_STATE = 5            # Seed for reproducibility
#     LABEL_NOISE = 0.02          # Add slight noise to test _apply_label_noise

#     # Calculate total clusters AFTER defining N_CLUSTERS_PER_CLASS
#     try:
#         num_total_clusters = sum(N_CLUSTERS_PER_CLASS) # Correctly calculate total clusters
#     except TypeError:
#         print("Error: N_CLUSTERS_PER_CLASS should be a list or tuple of integers.")
#         exit()


#     # Use a list for cluster_std as we have multiple clusters
#     # Length must match num_total_clusters (1 + 2 + 3 = 6)
#     CLUSTER_STDS = [0.4, 0.5, 0.7, 0.45, 0.65, 0.8] # Example stds for 6 clusters
#     if len(CLUSTER_STDS) != num_total_clusters:
#         print(f"Error: Length of CLUSTER_STDS ({len(CLUSTER_STDS)}) must match the total number of clusters ({num_total_clusters}).")
#         # Adjust CLUSTER_STDS or N_CLUSTERS_PER_CLASS
#         # Example fallback: Use a single float value
#         print("Fallback: Using a single float value (1.0) for cluster_std.")
#         CLUSTER_STDS = 1.0
#         # Or exit() if strict matching is required


#     datasets = {}
#     covariance_types = ['isotropic', 'diagonal', 'full']

#     # Generate datasets for each covariance type
#     for cov_type in covariance_types:
#         print(f"Generating dataset for covariance_type='{cov_type}'...")
#         try:
#             datasets[cov_type] = MoGSyntheticDataset(
#                 n_samples=N_SAMPLES,
#                 n_features=N_FEATURES,
#                 n_classes=N_CLASSES,
#                 n_clusters_per_class=N_CLUSTERS_PER_CLASS,
#                 class_sep=CLASS_SEP,  # Adjust separation as needed
#                 base_cluster_std=CLUSTER_STDS,
#                 intra_class_spread=INTRA_CLASS_SEP, # Use the correctly named variable
#                 covariance_type=cov_type,
#                 label_noise=LABEL_NOISE,
#                 random_state=RANDOM_STATE
#             )
#             print(f" -> Done. Features shape: {datasets[cov_type].features.shape}")
#         except ValueError as e:
#             print(f" -> Error generating dataset for {cov_type}: {e}")
#             # Handle error, maybe skip this cov_type or exit
#             datasets[cov_type] = None # Mark as failed


#     # Visualization
#     print("\nVisualizing datasets...")
#     # Filter out failed datasets
#     valid_cov_types = [ct for ct in covariance_types if datasets.get(ct) is not None]
#     if not valid_cov_types:
#          print("No valid datasets generated for visualization.")
#          exit()

#     fig, axes = plt.subplots(1, len(valid_cov_types), figsize=(6 * len(valid_cov_types), 6), sharex=True, sharey=True)
#     # Ensure axes is always iterable, even if only one subplot
#     if len(valid_cov_types) == 1:
#         axes = [axes]

#     fig.suptitle(f'Synthetic Datasets (N={N_SAMPLES}, K={N_CLASSES}, Clusters/Class={N_CLUSTERS_PER_CLASS})', fontsize=16)

#     # Define colors for classes
#     colors = plt.cm.viridis(np.linspace(0, 1, N_CLASSES))

#     plot_handles = [] # For custom legend
#     plot_labels = []  # For custom legend

#     for i, cov_type in enumerate(valid_cov_types):
#         ax = axes[i]
#         dataset = datasets[cov_type]

#         # Get data as numpy arrays
#         X = dataset.features.numpy()
#         y = dataset.labels.numpy()
#         means = dataset.cluster_means_.numpy() # Shape: (num_total_clusters, n_features)
#         # *** FIX: Use the dataset's mapping attribute ***
#         cluster_to_class_map = dataset.cluster_class_labels_.numpy() # Get the mapping

#         # Plot data points for each class
#         for k in range(N_CLASSES):
#             class_mask = (y == k)
#             scatter = ax.scatter(X[class_mask, 0], X[class_mask, 1],
#                        color=colors[k],
#                        # label=f'Class {k}', # Labeling inside loop creates duplicates
#                        s=10, alpha=0.7, edgecolors='w', linewidth=0.5)
#             # Store handle only once per class for the legend
#             if i == 0: # Only take handles from the first plot for the main legend
#                  plot_handles.append(scatter)
#                  plot_labels.append(f'Class {k}')


#         # Plot cluster means, using the correct class mapping
#         for cluster_idx in range(dataset.num_total_clusters):
#              # *** FIX: Get class label from the dataset's mapping ***
#              class_label = cluster_to_class_map[cluster_idx]
#              mean_marker = ax.scatter(means[cluster_idx, 0], means[cluster_idx, 1],
#                         color=colors[class_label], # Use mapped class label for color
#                         marker='X', s=150, edgecolors='black', linewidth=1.5)
#              # Add handle for mean marker only once for the legend
#              if i == 0 and cluster_idx == 0:
#                   plot_handles.append(mean_marker)
#                   plot_labels.append('Cluster Mean')


#         ax.set_title(f"Covariance: '{cov_type}'")
#         ax.set_xlabel("Feature 1")
#         if i == 0:
#             ax.set_ylabel("Feature 2")

#         ax.grid(True, linestyle='--', alpha=0.5)
#         # Set aspect ratio to equal to see shape distortion accurately
#         ax.set_aspect('equal', adjustable='box')

#     # Create a single legend for the entire figure
#     fig.legend(plot_handles, plot_labels, loc='upper right', bbox_to_anchor=(0.99, 0.95))

#     plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout further for global legend
#     plt.show()

#     print("\nVisualization complete.")