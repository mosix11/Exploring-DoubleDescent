import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np 
import math # To help determine subplot grid

# def analyze_and_plot_results_combined(root_directory):
#     """
#     Reads results from JSON files in subdirectories, extracts relevant metrics,
#     and plots both loss and accuracy figures in a combined plot.

#     Args:
#         root_directory (str): The path to the root directory containing the
#                               model-specific subdirectories.
#     """
#     results = {}
#     # subdirectory_pattern = re.compile(r"fc1\|h(\d+)_mnist\|ln0\.0\|noaug\|subsample\(4000, 1000\)_sgd\|lr0\.01\|b256\|noAMP\|step")
#     subdirectory_pattern = re.compile(r"cnn5\|k(\d+)_cifar10\|ln0\.2\|noaug\|full_seed11_sgd\|lr0\.01\|b128\|AMP\|isqrt")
#     subdirectory_pattern = re.compile(r"cnn5\|k(\d+)_cifar10")

#     for item in os.listdir(root_directory):
#         subdirectory_path = os.path.join(root_directory, item)
#         match = subdirectory_pattern.match(item)
#         if os.path.isdir(subdirectory_path) and match:
#             param = int(match.group(1))
#             log_file_path = os.path.join(subdirectory_path, "log", "results.json")
#             if os.path.exists(log_file_path):
#                 try:
#                     with open(log_file_path, 'r') as f:
#                         data = json.load(f)
#                         results[param] = {
#                             "final_train_loss": data.get("final", {}).get("Train/Loss"),
#                             "best_train_loss": data.get("best", {}).get("Train/Loss"),
#                             "final_test_loss": data.get("final", {}).get("Test/Loss"),
#                             "best_test_loss": data.get("best", {}).get("Test/Loss"),
#                             "final_train_acc": data.get("final", {}).get("Train/ACC"),
#                             "best_train_acc": data.get("best", {}).get("Train/ACC"),
#                             "final_test_acc": data.get("final", {}).get("Test/ACC"),
#                             "best_test_acc": data.get("best", {}).get("Test/ACC"),
#                         }
#                 except FileNotFoundError:
#                     print(f"Warning: {log_file_path} not found.")
#                 except json.JSONDecodeError:
#                     print(f"Warning: Could not decode JSON in {log_file_path}.")

#     if not results:
#         print(f"No matching directories or results found in {root_directory}.")
#         return

#     params = sorted(results.keys())

#     # Create a figure with two subplots
#     fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
#     fig.suptitle('Training Performance vs. Parameter (h)')

#     # Plotting Loss on the top subplot
#     axs[0].plot(params, [results[p]["final_train_loss"] for p in params], marker='o', label='Final Train Loss')
#     axs[0].plot(params, [results[p]["best_train_loss"] for p in params], marker='o', label='Best Train Loss')
#     axs[0].plot(params, [results[p]["final_test_loss"] for p in params], marker='o', label='Final Test Loss')
#     axs[0].plot(params, [results[p]["best_test_loss"] for p in params], marker='o', label='Best Test Loss')
#     axs[0].set_ylabel('Loss')
#     axs[0].set_title('Loss')
#     axs[0].legend()
#     axs[0].grid(True)

#     # Plotting Accuracy on the bottom subplot
#     axs[1].plot(params, [results[p]["final_train_acc"] for p in params], marker='o', label='Final Train Accuracy')
#     axs[1].plot(params, [results[p]["best_train_acc"] for p in params], marker='o', label='Best Train Accuracy')
#     axs[1].plot(params, [results[p]["final_test_acc"] for p in params], marker='o', label='Final Test Accuracy')
#     axs[1].plot(params, [results[p]["best_test_acc"] for p in params], marker='o', label='Best Test Accuracy')
#     axs[1].set_xlabel('Parameter (h)')
#     axs[1].set_ylabel('Accuracy')
#     axs[1].set_title('Accuracy')
#     axs[1].legend()
#     axs[1].grid(True)

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap
#     plt.savefig(os.path.join(root_directory, "combined_performance_vs_param.png"))
#     plt.show()

# if __name__ == "__main__":
#     # root_dir = "outputs/modelwise/FC1_MNIST(subsampe(4000, 1000)+NoAug+0.0Noise)_Parallel_Seed11"
#     root_dir = "outputs/modelwise/CNN5_CIFAR10+NoAug+0.2Noise_Parallel_Seed11"
#     analyze_and_plot_results_combined(root_dir)
#     print("Script finished. Combined plot saved as combined_performance_vs_param.png in the root directory.")


# --- Configuration ---
base_path_reuse = 'outputs/modelwise/FC1_MoG(smpls100000+ftrs512+cls30+0.2Noise)_WeightReuse_Seed22'
base_path_parallel = 'outputs/modelwise/FC1_MoG(smpls100000+ftrs512+cls30+0.2Noise)_Parallel_Seed22'
dir_pattern = re.compile(r"fc1\|h(\d+)_mog")
json_rel_path = 'log/results.json'

# Metrics to plot (key in the plot title, path within the JSON)
metrics_to_plot = {
    "Final Train Loss": ("final", "Train/Loss"),
    "Final Train ACC": ("final", "Train/ACC"),
    "Final Test Loss": ("final", "Test/Loss"),
    "Final Test ACC": ("final", "Test/ACC"),
    "Best Train Loss": ("best", "Train/Loss"),
    "Best Train ACC": ("best", "Train/ACC"),
    "Best Test Loss": ("best", "Test/Loss"),
    "Best Test ACC": ("best", "Test/ACC"),
}

# --- Function to load results (same as before) ---
def load_experiment_results(base_path, pattern, json_path):
    """
    Loads results from JSON files in directories matching the pattern.

    Args:
        base_path (str): The root directory for the experiment.
        pattern (re.Pattern): Compiled regex pattern to match subdirectories.
        json_path (str): Relative path to the JSON file within subdirectories.

    Returns:
        dict: A dictionary mapping the extracted integer (h) to the loaded JSON data.
              Returns None for h if JSON loading fails.
    """
    results = {}
    if not os.path.isdir(base_path):
        print(f"Warning: Base directory not found: {base_path}")
        return results

    for item_name in os.listdir(base_path):
        item_path = os.path.join(base_path, item_name)
        if os.path.isdir(item_path):
            match = pattern.match(item_name)
            if match:
                try:
                    h_value = int(match.group(1))
                    full_json_path = os.path.join(item_path, json_path)

                    if os.path.exists(full_json_path):
                        with open(full_json_path, 'r') as f:
                            try:
                                data = json.load(f)
                                results[h_value] = data
                            except json.JSONDecodeError:
                                print(f"Warning: Could not decode JSON in {full_json_path}")
                                results[h_value] = None # Mark as failed load
                    else:
                        print(f"Warning: JSON file not found: {full_json_path}")
                        results[h_value] = None # Mark as missing file

                except ValueError:
                    print(f"Warning: Could not parse integer from directory name {item_name}")
                except Exception as e:
                     print(f"Warning: An unexpected error occurred processing {item_path}: {e}")
                     # Attempt to store None even if h_value wasn't parsed correctly, maybe less ideal
                     # results[h_value] = None # Mark generic error - h_value might not be defined
                     pass # Or just skip this problematic directory


    return results

# --- Load Data ---
print("Loading WeightReuse results...")
results_reuse = load_experiment_results(base_path_reuse, dir_pattern, json_rel_path)
print(f"Found {len(results_reuse)} potential results for WeightReuse.")

print("\nLoading Parallel results...")
results_parallel = load_experiment_results(base_path_parallel, dir_pattern, json_rel_path)
print(f"Found {len(results_parallel)} potential results for Parallel.")

# --- Prepare Data for Plotting ---
valid_reuse_h = set(h for h, data in results_reuse.items() if data is not None)
valid_parallel_h = set(h for h, data in results_parallel.items() if data is not None)
common_h = sorted(list(valid_reuse_h & valid_parallel_h))


if not common_h:
    print("\nError: No common 'h' values found with valid results for both experiments. Cannot generate plots.")
else:
    print(f"\nFound {len(common_h)} common 'h' values with valid results: {common_h}")

    plot_data = {}
    metrics_with_data = [] # Keep track of metrics we actually have data for

    for metric_name, (json_key1, json_key2) in metrics_to_plot.items():
        current_metric_data = {
            'h': [],
            'reuse': [],
            'parallel': []
        }
        valid_points_for_metric = 0
        for h in common_h:
            try:
                # Double check data exists for this specific h and metric
                reuse_val = results_reuse[h][json_key1][json_key2]
                parallel_val = results_parallel[h][json_key1][json_key2]

                # Ensure values are numeric (or skip)
                if isinstance(reuse_val, (int, float)) and isinstance(parallel_val, (int, float)):
                    current_metric_data['h'].append(h)
                    current_metric_data['reuse'].append(reuse_val)
                    current_metric_data['parallel'].append(parallel_val)
                    valid_points_for_metric += 1
                else:
                     print(f"Warning: Skipping h={h} for metric '{metric_name}'. Non-numeric data found.")

            except (KeyError, TypeError) as e:
                # This h value is skipped for this metric if data is missing/malformed
                # print(f"Debug: Skipping h={h} for metric '{metric_name}'. Error: {e}") # Optional debug print
                pass # Silently skip if keys don't exist for this h

        if valid_points_for_metric > 0:
             plot_data[metric_name] = current_metric_data
             metrics_with_data.append(metric_name)
        else:
            print(f"Warning: No valid comparable data points found for metric '{metric_name}'. It will not be plotted.")


    # --- Generate Combined Plot with Subplots ---
    num_metrics_to_plot = len(metrics_with_data)
    if num_metrics_to_plot > 0:
        print(f"\nGenerating combined plot for {num_metrics_to_plot} metrics...")

        # Determine grid size (aim for 2 columns)
        ncols = 2
        nrows = math.ceil(num_metrics_to_plot / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows), squeeze=False) # Adjust figsize as needed
        axes = axes.flatten() # Flatten the 2D array of axes for easy iteration

        plot_index = 0
        for metric_name in metrics_with_data: # Iterate only through metrics that have data
            if plot_index >= len(axes): # Should not happen with ceil, but safety check
                 break

            ax = axes[plot_index]
            data = plot_data[metric_name]

            ax.plot(data['h'], data['parallel'], marker='o', linestyle='-', label='Random Init')
            ax.plot(data['h'], data['reuse'], marker='x', linestyle='--', label='WeightReuse')

            ax.set_xscale('log') # Use log scale for h axis
            ax.set_xlabel('Hidden Dimension (h)')
            # Extract last part of metric name (Loss or ACC) for Y label
            y_label = metric_name.split()[-1] if ' ' in metric_name else metric_name
            ax.set_ylabel(y_label)
            ax.set_title(metric_name) # Use the full metric name as title
            ax.legend()
            ax.grid(True, which="both", ls="--", alpha=0.6) # Add grid

            plot_index += 1

        # Turn off any unused subplots if the number of metrics isn't a perfect multiple of ncols
        for i in range(plot_index, len(axes)):
            axes[i].axis('off')

        plt.tight_layout() # Adjust layout to prevent overlap
        print("\nDisplaying combined plot...")
        plt.show()
        print("Done.")

    else:
        print("\nError: No metrics had valid data points across common 'h' values. No plot generated.")
