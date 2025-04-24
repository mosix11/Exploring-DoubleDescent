import os
import re
import json
import matplotlib.pyplot as plt

def analyze_and_plot_results_combined(root_directory):
    """
    Reads results from JSON files in subdirectories, extracts relevant metrics,
    and plots both loss and accuracy figures in a combined plot.

    Args:
        root_directory (str): The path to the root directory containing the
                              model-specific subdirectories.
    """
    results = {}
    # subdirectory_pattern = re.compile(r"fc1\|h(\d+)_mnist\|ln0\.0\|noaug\|subsample\(4000, 1000\)_sgd\|lr0\.01\|b256\|noAMP\|step")
    subdirectory_pattern = re.compile(r"cnn5\|k(\d+)_cifar10\|ln0\.2\|noaug\|full_seed11_sgd\|lr0\.01\|b128\|AMP\|isqrt")

    for item in os.listdir(root_directory):
        subdirectory_path = os.path.join(root_directory, item)
        match = subdirectory_pattern.match(item)
        if os.path.isdir(subdirectory_path) and match:
            param = int(match.group(1))
            log_file_path = os.path.join(subdirectory_path, "log", "results.json")
            if os.path.exists(log_file_path):
                try:
                    with open(log_file_path, 'r') as f:
                        data = json.load(f)
                        results[param] = {
                            "final_train_loss": data.get("final", {}).get("Train/Loss"),
                            "best_train_loss": data.get("best", {}).get("Train/Loss"),
                            "final_test_loss": data.get("final", {}).get("Test/Loss"),
                            "best_test_loss": data.get("best", {}).get("Test/Loss"),
                            "final_train_acc": data.get("final", {}).get("Train/ACC"),
                            "best_train_acc": data.get("best", {}).get("Train/ACC"),
                            "final_test_acc": data.get("final", {}).get("Test/ACC"),
                            "best_test_acc": data.get("best", {}).get("Test/ACC"),
                        }
                except FileNotFoundError:
                    print(f"Warning: {log_file_path} not found.")
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON in {log_file_path}.")

    if not results:
        print(f"No matching directories or results found in {root_directory}.")
        return

    params = sorted(results.keys())

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Training Performance vs. Parameter (h)')

    # Plotting Loss on the top subplot
    axs[0].plot(params, [results[p]["final_train_loss"] for p in params], marker='o', label='Final Train Loss')
    axs[0].plot(params, [results[p]["best_train_loss"] for p in params], marker='o', label='Best Train Loss')
    axs[0].plot(params, [results[p]["final_test_loss"] for p in params], marker='o', label='Final Test Loss')
    axs[0].plot(params, [results[p]["best_test_loss"] for p in params], marker='o', label='Best Test Loss')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plotting Accuracy on the bottom subplot
    axs[1].plot(params, [results[p]["final_train_acc"] for p in params], marker='o', label='Final Train Accuracy')
    axs[1].plot(params, [results[p]["best_train_acc"] for p in params], marker='o', label='Best Train Accuracy')
    axs[1].plot(params, [results[p]["final_test_acc"] for p in params], marker='o', label='Final Test Accuracy')
    axs[1].plot(params, [results[p]["best_test_acc"] for p in params], marker='o', label='Best Test Accuracy')
    axs[1].set_xlabel('Parameter (h)')
    axs[1].set_ylabel('Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent overlap
    plt.savefig(os.path.join(root_directory, "combined_performance_vs_param.png"))
    plt.show()

if __name__ == "__main__":
    # root_dir = "outputs/modelwise/FC1_MNIST(subsampe(4000, 1000)+NoAug+0.0Noise)_Parallel_Seed11"
    root_dir = "outputs/modelwise/CNN5_CIFAR10+NoAug+0.2Noise_Parallel_Seed11"
    analyze_and_plot_results_combined(root_dir)
    print("Script finished. Combined plot saved as combined_performance_vs_param.png in the root directory.")