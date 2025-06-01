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
base_path_reuse_freeze = 'outputs/modelwise/FC1_MoG(smpls100000+ftrs512+cls30+0.2Noise)_WeightReuseFreeze_Seed22'

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

# --- Function to load results ---
def load_experiment_results(base_path, pattern, json_path):
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
                                results[h_value] = None
                    else:
                        print(f"Warning: JSON file not found: {full_json_path}")
                        results[h_value] = None
                except Exception as e:
                    print(f"Warning: Skipping {item_path} due to error: {e}")
    return results

# --- Load Data ---
print("Loading WeightReuse results...")
results_reuse = load_experiment_results(base_path_reuse, dir_pattern, json_rel_path)
print(f"Found {len(results_reuse)} potential results for WeightReuse.")

print("\nLoading Parallel results...")
results_parallel = load_experiment_results(base_path_parallel, dir_pattern, json_rel_path)
print(f"Found {len(results_parallel)} potential results for Parallel.")

print("\nLoading WeightReuse+Freeze results...")
results_reuse_freeze = load_experiment_results(base_path_reuse_freeze, dir_pattern, json_rel_path)
print(f"Found {len(results_reuse_freeze)} potential results for WeightReuse+Freeze.")

# --- Prepare Data ---
valid_reuse_h = set(h for h, data in results_reuse.items() if data is not None)
valid_parallel_h = set(h for h, data in results_parallel.items() if data is not None)
valid_freeze_h = set(h for h, data in results_reuse_freeze.items() if data is not None)
common_h = sorted(list(valid_reuse_h & valid_parallel_h & valid_freeze_h))

if not common_h:
    print("\nError: No common 'h' values found across all three experiments. Cannot generate plots.")
else:
    print(f"\nFound {len(common_h)} common 'h' values: {common_h}")

    plot_data = {}
    metrics_with_data = []

    for metric_name, (json_key1, json_key2) in metrics_to_plot.items():
        current_metric_data = {
            'h': [],
            'reuse': [],
            'parallel': [],
            'freeze': []
        }
        valid_points_for_metric = 0
        for h in common_h:
            try:
                reuse_val = results_reuse[h][json_key1][json_key2]
                parallel_val = results_parallel[h][json_key1][json_key2]
                freeze_val = results_reuse_freeze[h][json_key1][json_key2]

                if all(isinstance(val, (int, float)) for val in [reuse_val, parallel_val, freeze_val]):
                    current_metric_data['h'].append(h)
                    current_metric_data['reuse'].append(reuse_val)
                    current_metric_data['parallel'].append(parallel_val)
                    current_metric_data['freeze'].append(freeze_val)
                    valid_points_for_metric += 1
                else:
                    print(f"Warning: Non-numeric value found for h={h} in metric '{metric_name}'")
            except (KeyError, TypeError):
                pass

        if valid_points_for_metric > 0:
            plot_data[metric_name] = current_metric_data
            metrics_with_data.append(metric_name)
        else:
            print(f"Warning: No valid data points for metric '{metric_name}'. Skipping it.")

    # --- Plot ---
    num_metrics = len(metrics_with_data)
    if num_metrics > 0:
        print(f"\nGenerating combined plot for {num_metrics} metrics...")
        ncols = 2
        nrows = math.ceil(num_metrics / ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4 * nrows), squeeze=False)
        axes = axes.flatten()

        for idx, metric_name in enumerate(metrics_with_data):
            ax = axes[idx]
            data = plot_data[metric_name]
            ax.plot(data['h'], data['parallel'], marker='o', linestyle='-', label='Random Init')
            ax.plot(data['h'], data['reuse'], marker='x', linestyle='--', label='WeightReuse')
            ax.plot(data['h'], data['freeze'], marker='s', linestyle='-.', label='WeightReuse+Freeze')

            ax.set_xscale('log')
            ax.set_xlabel('Hidden Dimension (h)')
            ax.set_ylabel(metric_name.split()[-1])
            ax.set_title(metric_name)
            ax.grid(True, which="both", ls="--", alpha=0.6)
            ax.legend()

        for i in range(num_metrics, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        print("\nDisplaying combined plot...")
        plt.show()
    else:
        print("\nError: No valid metrics with data to plot.")