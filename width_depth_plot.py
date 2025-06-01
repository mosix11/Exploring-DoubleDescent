import os
import json
import re
import matplotlib.pyplot as plt

def plot_double_descent_curves(base_dir):
    """
    Plots the double descent curves for different network depths based on
    results stored in subdirectories.

    Args:
        base_dir (str): The path to the base directory containing the experiment results.
    """
    
    # Define the metrics to plot
    metrics_to_plot = {
        "final": ["Train/Loss", "Train/ACC", "Test/Loss", "Test/ACC"],
        "best": ["Train/Loss", "Train/ACC", "Test/Loss", "Test/ACC"]
    }

    # Dictionary to store data for plotting:
    # { num_hidden_layers: { param_count: { "final": {...}, "best": {...} } } }
    all_data = {}

    # Regex to extract num_hidden_layers (N) and parameter count (P)
    dir_pattern = re.compile(r"fc(\d+)\|h.*\|p(\d+)")

    for root, dirs, files in os.walk(base_dir):
        if "results.json" in files and os.path.basename(root) == "log":
            parent_dir = os.path.dirname(root) # Go up from 'log' to the experiment directory
            match = dir_pattern.match(os.path.basename(parent_dir))
            
            if match:
                num_hidden_layers = int(match.group(1))
                param_count = int(match.group(2))
                
                results_file_path = os.path.join(root, "results.json")
                try:
                    with open(results_file_path, 'r') as f:
                        results = json.load(f)
                        
                        if num_hidden_layers not in all_data:
                            all_data[num_hidden_layers] = {}
                        
                        all_data[num_hidden_layers][param_count] = results
                except Exception as e:
                    print(f"Error reading {results_file_path}: {e}")
            else:
                print(f"Skipping directory (no match): {os.path.basename(parent_dir)}")

    # Sort the hidden layer depths for consistent plotting order
    sorted_depths = sorted(all_data.keys())

    # Create the 8 plots
    fig, axes = plt.subplots(2, 4, figsize=(24, 12)) # 2 rows, 4 columns
    axes = axes.flatten() # Flatten the 2x4 array for easy iteration

    plot_idx = 0
    for category, metrics in metrics_to_plot.items():
        for metric in metrics:
            ax = axes[plot_idx]
            ax.set_title(f"{category.capitalize()} - {metric}")
            ax.set_xlabel("Number of Parameters")
            ax.set_ylabel(metric)

            # Set both x and y axes to log scale
            ax.set_xscale('log')
            # ax.set_yscale('log')

            for depth in sorted_depths:
                x_values = []
                y_values = []
                # Sort by parameter count to ensure lines are drawn correctly
                sorted_param_counts = sorted(all_data[depth].keys())
                for param_count in sorted_param_counts:
                    if category in all_data[depth][param_count] and \
                       metric in all_data[depth][param_count][category]:
                        x_values.append(param_count)
                        y_values.append(all_data[depth][param_count][category][metric])
                
                if x_values and y_values: # Only plot if there's data
                    ax.plot(x_values, y_values, label=f'Depth {depth}')
            
            ax.legend(title="Hidden Layers")
            ax.grid(True)
            plot_idx += 1

    plt.tight_layout()
    plt.suptitle("Effect of Model Depth on Double Descent Curves (Log Scale)", y=1.02, fontsize=16)
    plt.show()

if __name__ == "__main__":
    base_directory = "outputs/width_depth/FC_WD_MoG(smpls100000+ftrs512+cls30+0.2Noise)_Parallel_B1024_Seed22"
    plot_double_descent_curves(base_directory)