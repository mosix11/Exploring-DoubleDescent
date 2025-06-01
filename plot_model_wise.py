import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np 
import math # To help determine subplot grid
from pathlib import Path
import pickle
import argparse
import ast








# def parse_extracted_value(value_str):
#     """
#     Parses a string that could be a single integer or a tuple string.

#     Args:
#         value_str (str): The string extracted by regex.

#     Returns:
#         int or tuple: The parsed integer or tuple of integers.
#     """
#     if value_str.startswith('(') and value_str.endswith(')'):
#         # It's a tuple-like string
#         try:
#             # ast.literal_eval is safer than eval() for parsing string literals
#             # as it only evaluates Python literal structures (numbers, strings, tuples, lists, dicts, booleans, None).
#             return ast.literal_eval(value_str)
#         except (ValueError, SyntaxError):
#             print(f"Warning: Could not parse '{value_str}' as a tuple. Returning as string.")
#             return value_str
#     else:
#         # It's a single integer string
#         try:
#             return int(value_str)
#         except ValueError:
#             print(f"Warning: Could not parse '{value_str}' as an integer. Returning as string.")
#             return value_str




def plot_modelwise_dd_curves(results, x_axis='p', x_log=True, smoothing_window=0):
    """
    Plots training and testing losses and accuracies from a dictionary with enhanced
    color and marker differentiation and optional data smoothing.

    Args:
        results (dict): A dictionary where keys are integer values and each
                        value is a dictionary with "final" and "best" keys.
                        Each of "final" and "best" contains "Train/Loss",
                        "Train/ACC", "Test/Loss", and "Test/ACC" with float values.
        x_axis (str, optional): The label for the x-axis. Defaults to 'p'.
        x_log (bool, optional): If True, sets the x-axis to a logarithmic scale.
                                Defaults to True.
        smoothing_window (int, optional): The size of the moving average window for
                                          smoothing the curves. If 0 or 1, no smoothing
                                          is applied. A larger value results in more smoothing.
                                          Defaults to 0.
    """

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
    
    # Prepare data for plotting
    x_values = sorted(results.keys())
    
    # Initialize dictionaries to store y-values for each metric
    plot_data = {label: [] for label in metrics_to_plot.keys()}
    
    for x_val in x_values:
        for label, (level1_key, level2_key) in metrics_to_plot.items():
            plot_data[label].append(results[x_val][level1_key][level2_key])

    # Apply smoothing if smoothing_window > 1
    if smoothing_window > 1:
        # Create a new dictionary to store smoothed data
        smoothed_plot_data = {label: [] for label in metrics_to_plot.keys()}
        for label, data in plot_data.items():
            # Convert list to numpy array for easier rolling mean calculation
            data_array = np.array(data)
            
            # Pad the data to handle edges for rolling mean without shortening the array
            # This is a simple padding; for more sophisticated edge handling, consider
            # scipy.ndimage.uniform_filter or pandas.Series.rolling.mean with min_periods
            padded_data = np.pad(data_array, (smoothing_window // 2, smoothing_window - 1 - smoothing_window // 2), mode='edge')
            
            # Calculate rolling mean using a convolution for efficiency
            weights = np.ones(smoothing_window) / smoothing_window
            smoothed_data = np.convolve(padded_data, weights, mode='valid')
            
            smoothed_plot_data[label] = smoothed_data.tolist()
        
        # Replace original plot_data with smoothed data
        plot_data = smoothed_plot_data

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    title_suffix = ""
    if smoothing_window > 1:
        title_suffix = f" (Smoothed with Window = {smoothing_window})"
    
    fig.suptitle(f'Model Performance Across Different {x_axis} Values{title_suffix}', fontsize=16)

    # Define distinct colors for each of the four curves
    curve_colors = {
        "Final Train": '#1f77b4',  # Matplotlib default blue
        "Final Test": '#ff7f0e',   # Matplotlib default orange
        "Best Train": '#2ca02c',   # Matplotlib default green
        "Best Test": '#d62728',    # Matplotlib default red
    }
    
    # Linestyles to differentiate "final" vs "best"
    linestyles = {
        "final": '-',  # Solid line for final
        "best": '--', # Dashed line for best
    }
    
    # Markers to differentiate within each plot (e.g., Train vs Test)
    markers = {
        ("final", "Train"): 'o',
        ("final", "Test"): 's',
        ("best", "Train"): '^',
        ("best", "Test"): 'D',
    }

    # Plot Losses (Left Plot)
    if x_log:
        ax1.set_xscale('log')
    ax1.set_title('Loss Comparison')
    ax1.set_xlabel(x_axis)
    ax1.set_ylabel('Loss Value')
    ax1.grid(True, which="both", ls="--", c='0.7', alpha=0.6)

    for label in ["Final Train Loss", "Final Test Loss", "Best Train Loss", "Best Test Loss"]:
        level1_key, metric_path = metrics_to_plot[label]
        data_type = "Train" if "Train" in label else "Test"
        
        color_key = f"{level1_key.capitalize()} {data_type}"
        marker_key = (level1_key, data_type)

        ax1.plot(x_values, plot_data[label],
                 label=label,
                 color=curve_colors[color_key],
                 linestyle=linestyles[level1_key],
                 marker=markers[marker_key],
                 markersize=6,
                 alpha=0.9)
    ax1.legend(loc='upper right')

    # Plot Accuracies (Right Plot)
    if x_log:
        ax2.set_xscale('log')
    ax2.set_title('Accuracy Comparison')
    ax2.set_xlabel(x_axis)
    ax2.set_ylabel('Accuracy Value')
    ax2.grid(True, which="both", ls="--", c='0.7', alpha=0.6)
    
    for label in ["Final Train ACC", "Final Test ACC", "Best Train ACC", "Best Test ACC"]:
        level1_key, metric_path = metrics_to_plot[label]
        data_type = "Train" if "Train" in label else "Test"

        color_key = f"{level1_key.capitalize()} {data_type}"
        marker_key = (level1_key, data_type)

        ax2.plot(x_values, plot_data[label],
                 label=label,
                 color=curve_colors[color_key],
                 linestyle=linestyles[level1_key],
                 marker=markers[marker_key],
                 markersize=6,
                 alpha=0.9)
    ax2.legend(loc='lower right')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    
    

def load_experiment_results(base_dir, pattern, key_group=1, json_path=Path('log/results.json')):
    results = {}
    if not os.path.isdir(base_dir):
        print(f"Warning: Base directory not found: {base_dir}")
        return results

    for item_name in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item_name)
        if os.path.isdir(item_path):
            match = pattern.match(item_name)
            if match:
                try:
                    key_value = match.group(key_group)
                    try:
                        key_value = int(key_value)
                    except Exception as e:
                        print(f"Warning: {item_path} match cannot be converted to integer error: {e}")
                    finally:
                        pass
                    
                
                    full_json_path = os.path.join(item_path, json_path)

                    if os.path.exists(full_json_path):
                        with open(full_json_path, 'r') as f:
                            try:
                                data = json.load(f)
                                results[key_value] = data
                            except json.JSONDecodeError:
                                print(f"Warning: Could not decode JSON in {full_json_path}")
                                results[key_value] = None
                    else:
                        print(f"Warning: JSON file not found: {full_json_path}")
                        results[key_value] = None
                except Exception as e:
                    print(f"Warning: Skipping {item_path} due to error: {e}")
                    
    return results


if __name__ == '__main__':
    
    base_dir = 'outputs/modelwise/FC1_MNIST(subsampe(4000, 1000)+NoAug+0.2Noise)_Parallel_Seed11'
    # base_dir = 'outputs/width_depth/FC_WD_MoG(smpls100000+ftrs512+cls30+0.2Noise)_Parallel_B1024_Seed22'
    
    json_rel_path = 'log/results.json'

    

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        help="The model to use for training.",
        type=str,
        required=True,
    )
    # parser.add_argument(
    #     "-d",
    #     "--dataset",
    #     help="The dataset used for trainig the model.",
    #     type=str,
    #     choices=["mnist", "cifar10", "cifar100", "mog"],
    #     required=True,
    # )
    parser.add_argument(
        "-x",
        "--xaxis",
        help="What to show on x axis",
        type=str,
        choices=["h", "p", "k"],
        required=True,
    )
    args = parser.parse_args()

    if args.xaxis == 'p':
        # dir_pattern = re.compile(r"fc(\d+)\|h.*\|p(\d+)")
        dir_pattern = re.compile(fr"{args.model}\|.*\|p(\d+)")
    else:
        first_value_pattern = r"(\d+|\(\d+(?:,\s*\d+)*\))"
        pattern_string = fr"{args.model}\|{args.xaxis}{first_value_pattern}\|p(\d+)"
        dir_pattern =  re.compile(pattern_string)
    
    
    results = load_experiment_results(base_dir=base_dir, pattern=dir_pattern, json_path=json_rel_path)
    
    x_axis = 'Parameters' if args.xaxis == 'p' else 'Width'
    plot_modelwise_dd_curves(results, x_axis=x_axis, x_log=True)
    
    # for s in test_strings:
    #     print(f"\nProcessing: '{s}'")
    #     match = dir_pattern.match(s)

    #     if match:
    #         extracted_first_value_str = match.group(1)
    #         extracted_p_value = int(match.group(2))

    #         parsed_first_value = parse_extracted_value(extracted_first_value_str)

    #         print(f"  Full match: {match.group(0)}")
    #         print(f"  Parsed first value: {parsed_first_value} (type: {type(parsed_first_value)})")
    #         print(f"  Second captured group (p value): {extracted_p_value}")
    #     else:
    #         print("  No match.")

        
    # dir_pattern = re.compile(r"fc1\|h(\d+)_mog")
    # dir_pattern = re.compile(r"fc1\|h(\d+)_mnist")
    # dir_pattern = re.compile(r"fc(\d+)\|h.*\|p(\d+)") # FCN
    # dir_pattern = re.compile(r"fc1\|h(\d+)\|p(\d+)") # FC1    



# h_values = []
# train_losses = []
# test_losses = []
# train_accuracies = []
# test_accuracies = []

# # results_dir = Path('outputs/results/fc1_mnist')
# # results_dir = Path('outputs/results/fc1_cifar10')
# results_dir = Path('outputs/results/cnn5_cifar10')

# for filename in os.listdir(results_dir):
#     match = re.match(r"res_param(\d+)\.pkl", filename)
#     if match:
#         h = int(match.group(1))
#         with open(results_dir / filename, "rb") as f:
#             data = pickle.load(f)
#         h_values.append(h)
#         train_losses.append(data["train_loss"])
#         test_losses.append(data["test_loss"])
#         train_accuracies.append(data["train_acc"])
#         test_accuracies.append(data["test_acc"])

# sorted_data = sorted(zip(h_values, train_losses, test_losses, train_accuracies, test_accuracies))
# h_values, train_losses, test_losses, train_accuracies, test_accuracies = zip(*sorted_data)

# fig, axs = plt.subplots(2, 1, figsize=(8, 10))

# axs[0].plot(h_values, train_losses, 'o-r', label='train')
# axs[0].plot(h_values, test_losses, 'o-b', label='test')
# axs[0].set_xlabel("Hidden layer size (h)")
# # axs[0].set_xlabel("CNN Width Parameter (K)")
# axs[0].set_ylabel("loss")
# # axs[0].set_xscale('log') 
# axs[0].legend()
# axs[0].grid(True)

# axs[1].plot(h_values, train_accuracies, 'o-r', label='train')
# axs[1].plot(h_values, test_accuracies, 'o-b', label='test')
# # axs[1].set_xlabel("Hidden layer size (h)")
# axs[1].set_xlabel("CNN Width Parameter (K)")
# axs[1].set_ylabel("accuracy")
# # axs[1].set_xscale('log') 
# axs[1].legend()
# axs[1].grid(True)

# plt.tight_layout()
# plt.show()
