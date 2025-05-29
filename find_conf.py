import collections
import math
import numpy as np
import statistics 
from tqdm import tqdm
import pickle

# --- Helper Functions ---

def _calculate_stdev(data: list[int]) -> float:
    """Calculates population standard deviation. Fallback if statistics.stdev is an issue."""
    if not data or len(data) < 2:
        return 0.0
    n = len(data)
    if n == 0: return 0.0 # Should be caught by len(data) < 2
    mean = sum(data) / n
    variance = sum([(x - mean) ** 2 for x in data]) / n 
    return math.sqrt(variance)

def calculate_balance_score(widths: list[int]) -> float:
    """Calculates balance score (standard deviation of widths). Lower is better."""
    if not widths or len(widths) < 2: # For 0 or 1 hidden layer, it's perfectly 'balanced'
        return 0.0
    try:
        # Use statistics.stdev if available and appropriate (it calculates sample stdev by default)
        # For just comparing, either population or sample is fine if used consistently.
        # Let's use population stdev for simplicity if implementing manually.
        return _calculate_stdev(widths) 
        # Alternatively, if you are sure statistics module is available and you want sample stdev:
        # return statistics.stdev(map(float, widths)) # Ensure elements are float for stdev
    except AttributeError: # Fallback if statistics.stdev is not found (older Python?)
        return _calculate_stdev(widths)
    except Exception: # General fallback
        return _calculate_stdev(widths)


def _calculate_params(input_dim: int, output_dim: int, hidden_widths: list[int], count_bias: bool) -> int | float:
    params = 0
    all_dims = [input_dim] + hidden_widths + [output_dim]
    for i in range(len(all_dims) - 1):
        dim_in = all_dims[i]
        dim_out = all_dims[i+1]
        if dim_in <= 0 or dim_out <= 0: return float('inf') 
        params += dim_in * dim_out
        if count_bias: params += dim_out
    return params

def _solve_for_last_hidden_width(
    p_anchor: int, input_dim: int, output_dim: int, count_bias: bool,
    prefix_hidden_widths: list[int] 
) -> int | None:
    params_prefix_layers = 0
    if not prefix_hidden_widths: 
        width_of_L_minus_1_layer = input_dim
    else:
        current_dims_for_prefix = [input_dim] + prefix_hidden_widths
        for i in range(len(current_dims_for_prefix) - 1):
            d_in, d_out = current_dims_for_prefix[i], current_dims_for_prefix[i+1]
            params_prefix_layers += d_in * d_out
            if count_bias: params_prefix_layers += d_out
        width_of_L_minus_1_layer = prefix_hidden_widths[-1]

    numerator = p_anchor - params_prefix_layers
    denominator = width_of_L_minus_1_layer + output_dim
    if count_bias:
        numerator -= output_dim 
        denominator += 1       
    if denominator <= 0: return None

    wL_float = numerator / denominator
    wL_candidates = []
    floor_wL, ceil_wL = math.floor(wL_float), math.ceil(wL_float)
    if floor_wL >= 1: wL_candidates.append(floor_wL)
    if ceil_wL >= 1 and ceil_wL != floor_wL: wL_candidates.append(ceil_wL)
    if not wL_candidates: return None

    best_wL_for_this_prefix, min_diff_for_this_prefix = None, float('inf')
    for wL_cand in wL_candidates:
        current_total_hidden_widths = prefix_hidden_widths + [wL_cand]
        actual_total_params = _calculate_params(input_dim, output_dim, current_total_hidden_widths, count_bias)
        if actual_total_params == float('inf'): continue
        diff = abs(actual_total_params - p_anchor)
        if diff < min_diff_for_this_prefix:
            min_diff_for_this_prefix, best_wL_for_this_prefix = diff, wL_cand
        elif diff == min_diff_for_this_prefix and (best_wL_for_this_prefix is None or wL_cand < best_wL_for_this_prefix):
            best_wL_for_this_prefix = wL_cand
    return best_wL_for_this_prefix

def _find_configs_for_L_hidden_layers(
    num_total_hidden_layers: int, 
    p_anchor: int, 
    k_anchor_1hl_width: int, 
    input_dim: int, 
    output_dim: int, 
    count_bias: bool,
    balance_param_delta_allowance: int 
) -> tuple[list[int] | None, list[int] | None]: # (closest_p_config, balanced_config)
    
    L = num_total_hidden_layers
    best_overall_widths_closest_p = None
    min_overall_diff_closest_p = float('inf')
    all_generated_valid_configs = [] # Store (widths, p_actual, diff_from_p_anchor)

    def _update_closest_p_config(new_widths_cand, p_actual_cand):
        nonlocal best_overall_widths_closest_p, min_overall_diff_closest_p
        if new_widths_cand is None or p_actual_cand == float('inf'): return
        current_diff = abs(p_actual_cand - p_anchor)
        if current_diff < min_overall_diff_closest_p:
            min_overall_diff_closest_p = current_diff
            best_overall_widths_closest_p = new_widths_cand
        elif current_diff == min_overall_diff_closest_p:
            if best_overall_widths_closest_p is None or \
               sum(new_widths_cand) < sum(best_overall_widths_closest_p) or \
               (sum(new_widths_cand) == sum(best_overall_widths_closest_p) and \
                new_widths_cand < best_overall_widths_closest_p):
                best_overall_widths_closest_p = new_widths_cand
    
    # --- Loops to generate configurations & find closest_p ---
    if L == 0: 
        current_widths = []
        actual_params = _calculate_params(input_dim, output_dim, current_widths, count_bias)
        if actual_params != float('inf'):
            _update_closest_p_config(current_widths, actual_params)
            all_generated_valid_configs.append((current_widths, actual_params, abs(actual_params - p_anchor)))
    elif L == 1: 
        w1_solved = _solve_for_last_hidden_width(p_anchor, input_dim, output_dim, count_bias, [])
        if w1_solved is not None:
            current_widths = [w1_solved]
            actual_params = _calculate_params(input_dim, output_dim, current_widths, count_bias)
            if actual_params != float('inf'):
                _update_closest_p_config(current_widths, actual_params)
                all_generated_valid_configs.append((current_widths, actual_params, abs(actual_params - p_anchor)))
    elif L == 2: 
        for w1 in range(1, k_anchor_1hl_width + 1):
            w2_solved = _solve_for_last_hidden_width(p_anchor, input_dim, output_dim, count_bias, [w1])
            if w2_solved is not None:
                current_widths = [w1, w2_solved]
                actual_params = _calculate_params(input_dim, output_dim, current_widths, count_bias)
                if actual_params != float('inf'):
                    _update_closest_p_config(current_widths, actual_params)
                    all_generated_valid_configs.append((current_widths, actual_params, abs(actual_params - p_anchor)))
    elif L == 3: 
        for w1 in range(1, k_anchor_1hl_width + 1):
            for w2 in range(1, k_anchor_1hl_width + 1): 
                w3_solved = _solve_for_last_hidden_width(p_anchor, input_dim, output_dim, count_bias, [w1, w2])
                if w3_solved is not None:
                    current_widths = [w1, w2, w3_solved]
                    actual_params = _calculate_params(input_dim, output_dim, current_widths, count_bias)
                    if actual_params != float('inf'):
                        _update_closest_p_config(current_widths, actual_params)
                        all_generated_valid_configs.append((current_widths, actual_params, abs(actual_params - p_anchor)))
    else:
        print(f"Warning: Config search for {L} hidden layers not implemented beyond L=3.")
        return None, None

    # --- Determine the "balanced_close_p" configuration ---
    best_overall_widths_balanced = None
    if not all_generated_valid_configs:
        return best_overall_widths_closest_p, best_overall_widths_closest_p if best_overall_widths_closest_p else None

    param_diff_threshold_for_balance = float('inf')
    if min_overall_diff_closest_p != float('inf'): # Check if any closest_p was found
        param_diff_threshold_for_balance = min_overall_diff_closest_p + balance_param_delta_allowance
    
    candidate_configs_for_balance_metric = []
    for cfg_widths, cfg_p_actual, cfg_diff in all_generated_valid_configs:
        if cfg_diff <= param_diff_threshold_for_balance:
            balance_score = calculate_balance_score(cfg_widths)
            candidate_configs_for_balance_metric.append({
                'widths': cfg_widths, 'diff': cfg_diff, 
                'balance_score': balance_score, 'sum_widths': sum(cfg_widths) if cfg_widths else 0
            })

    if not candidate_configs_for_balance_metric:
        best_overall_widths_balanced = best_overall_widths_closest_p
    else:
        candidate_configs_for_balance_metric.sort(key=lambda x: (
            x['balance_score'], x['diff'], x['sum_widths'], x['widths']
        ))
        best_overall_widths_balanced = candidate_configs_for_balance_metric[0]['widths']

    return best_overall_widths_closest_p, best_overall_widths_balanced

# --- Main Function ---

def get_network_configurations_range(
    input_dim: int, 
    output_dim: int, 
    target_hidden_layers: list[int], 
    count_bias: bool, 
    max_anchor_1hl_width: int,
    balance_param_delta_allowance: int = 5 # New parameter with default
) -> collections.OrderedDict:
    if not isinstance(input_dim, int) or input_dim < 1:
        raise ValueError("input_dim must be a positive integer.")
    # ... (add other input validations as in v2) ...
    if not isinstance(balance_param_delta_allowance, int) or balance_param_delta_allowance < 0:
        raise ValueError("balance_param_delta_allowance must be a non-negative integer.")


    result_dict = collections.OrderedDict()
    sorted_target_hl = sorted(list(set(target_hidden_layers)))

    for k_1hl in tqdm(range(1, max_anchor_1hl_width + 1)):
        anchor_1hl_widths = [k_1hl]
        p_anchor = _calculate_params(input_dim, output_dim, anchor_1hl_widths, count_bias)
        if p_anchor == float('inf'): continue

        current_group_configs = collections.OrderedDict()
        all_depths_found_for_this_anchor = True

        for L_target in sorted_target_hl:
            closest_p_cfg, balanced_cfg = None, None
            if L_target == 1: 
                params_check = _calculate_params(input_dim, output_dim, anchor_1hl_widths, count_bias)
                if params_check == p_anchor: # This should always be true
                    closest_p_cfg = anchor_1hl_widths
                    # For L=1, balanced is same as closest
                    balanced_cfg = anchor_1hl_widths 
                else: # Fallback, should ideally not be hit for L_target=1 if p_anchor is from k_1hl
                    closest_p_cfg, balanced_cfg = _find_configs_for_L_hidden_layers(
                        L_target, p_anchor, k_1hl, 
                        input_dim, output_dim, count_bias, balance_param_delta_allowance
                    )
            else: 
                closest_p_cfg, balanced_cfg = _find_configs_for_L_hidden_layers(
                    L_target, p_anchor, k_1hl, 
                    input_dim, output_dim, count_bias, balance_param_delta_allowance
                )
            
            if closest_p_cfg is not None and balanced_cfg is not None : # Both must be found
                current_group_configs[L_target] = {
                    'closest_p': closest_p_cfg,
                    'balanced_close_p': balanced_cfg
                }
            else:
                all_depths_found_for_this_anchor = False
                break 
        
        if all_depths_found_for_this_anchor and len(current_group_configs) == len(sorted_target_hl):
            if p_anchor not in result_dict: 
                result_dict[p_anchor] = current_group_configs
    return result_dict


def get_network_configurations_list(
    input_dim: int, 
    output_dim: int, 
    target_hidden_layers: list[int], 
    count_bias: bool, 
    anchor_1hl_width_list: list[int],
    balance_param_delta_allowance: int = 5 # New parameter with default
) -> collections.OrderedDict:
    if not isinstance(input_dim, int) or input_dim < 1:
        raise ValueError("input_dim must be a positive integer.")
    # ... (add other input validations as in v2) ...
    if not isinstance(balance_param_delta_allowance, int) or balance_param_delta_allowance < 0:
        raise ValueError("balance_param_delta_allowance must be a non-negative integer.")


    result_dict = collections.OrderedDict()
    sorted_target_hl = sorted(list(set(target_hidden_layers)))

    for k_1hl in tqdm(anchor_1hl_width_list):
        anchor_1hl_widths = [k_1hl]
        p_anchor = _calculate_params(input_dim, output_dim, anchor_1hl_widths, count_bias)
        if p_anchor == float('inf'): continue

        current_group_configs = collections.OrderedDict()
        all_depths_found_for_this_anchor = True

        for L_target in sorted_target_hl:
            closest_p_cfg, balanced_cfg = None, None
            if L_target == 1: 
                params_check = _calculate_params(input_dim, output_dim, anchor_1hl_widths, count_bias)
                if params_check == p_anchor: # This should always be true
                    closest_p_cfg = anchor_1hl_widths
                    # For L=1, balanced is same as closest
                    balanced_cfg = anchor_1hl_widths 
                else: # Fallback, should ideally not be hit for L_target=1 if p_anchor is from k_1hl
                    closest_p_cfg, balanced_cfg = _find_configs_for_L_hidden_layers(
                        L_target, p_anchor, k_1hl, 
                        input_dim, output_dim, count_bias, balance_param_delta_allowance
                    )
            else: 
                closest_p_cfg, balanced_cfg = _find_configs_for_L_hidden_layers(
                    L_target, p_anchor, k_1hl, 
                    input_dim, output_dim, count_bias, balance_param_delta_allowance
                )
            
            if closest_p_cfg is not None and balanced_cfg is not None : # Both must be found
                current_group_configs[L_target] = {
                    'closest_p': closest_p_cfg,
                    'balanced_close_p': balanced_cfg
                }
            else:
                all_depths_found_for_this_anchor = False
                break 
        
        if all_depths_found_for_this_anchor and len(current_group_configs) == len(sorted_target_hl):
            if p_anchor not in result_dict: 
                result_dict[p_anchor] = current_group_configs
    return result_dict


# --- Example Usage ---

input_features = 512
output_classes = 30
layers_to_test = [1, 2, 3]
include_bias = True
max_k_for_1hl_anchor = 256

fc1_h_widths= [
    1,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
    34,
    36,
    38,
    40,
    42,
    44,
    46,
    48,
    50,
    52,
    56,
    60,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
    136,
    144,
    152,
    160,
    168,
    176,
    184,
    192,
    200,
    208,
    216,
    224,
    232,
    240,
    248,
    256,
    264,
    272,
    280,
    288,
    296,
    304,
    312,
    320,
    328,
    336,
    344,
    368,
    392,
    432,
    464,
    512,
    640,
    768,
    896,
    1024,
    2048,
    3072,
    4096,
    8192,
    16384
]

# New parameter: how much additional difference from P_anchor is allowed 
# when searching for a balanced configuration.
# If 0, balanced search only considers those configs that are *as close as* the closest_p one.
balance_delta = 30 



# print(f"Searching configurations with balance consideration:")
# print(f"  Input Dimensions: {input_features}")
# print(f"  Output Dimensions: {output_classes}")
# print(f"  Target Hidden Layers: {layers_to_test}")
# print(f"  Count Bias: {include_bias}")
# print(f"  Max Anchor (1-HL) Width: {max_k_for_1hl_anchor}")
# print(f"  Balance Parameter Delta Allowance: {balance_delta}\n")

# configurations_v3 = get_network_configurations_range(
#     input_dim=input_features,
#     output_dim=output_classes,
#     target_hidden_layers=layers_to_test,
#     count_bias=include_bias,
#     max_anchor_1hl_width=max_k_for_1hl_anchor,
#     balance_param_delta_allowance=balance_delta
# )

# if not configurations_v3:
#     print("No matching configuration groups found.")
# else:
#     print(f"Found {len(configurations_v3)} groups:")
#     for p_anchor_key, group_data in configurations_v3.items():
#         print(f"\n--- Anchor (1-HL) Params Target: {p_anchor_key} ---")
#         for num_hl, configs in group_data.items():
#             print(f"  {num_hl} Hidden Layer(s):")
            
#             # Closest P Config
#             widths_closest = configs['closest_p']
#             params_closest = _calculate_params(input_features, output_classes, widths_closest, include_bias)
#             diff_closest = params_closest - p_anchor_key if params_closest != float('inf') else "N/A"
#             balance_closest = calculate_balance_score(widths_closest)
#             print(f"    Closest P:       Widths={widths_closest}, Params={params_closest}, Diff={diff_closest}, Balance (StdDev)={balance_closest:.2f}")

#             # Balanced Close P Config
#             widths_balanced = configs['balanced_close_p']
#             params_balanced = _calculate_params(input_features, output_classes, widths_balanced, include_bias)
#             diff_balanced = params_balanced - p_anchor_key if params_balanced != float('inf') else "N/A"
#             balance_balanced = calculate_balance_score(widths_balanced)
#             print(f"    Balanced Close P: Widths={widths_balanced}, Params={params_balanced}, Diff={diff_balanced}, Balance (StdDev)={balance_balanced:.2f}")



configurations = get_network_configurations_list(
    input_dim=input_features,
    output_dim=output_classes,
    target_hidden_layers=layers_to_test,
    count_bias=include_bias,
    anchor_1hl_width_list=fc1_h_widths,
    balance_param_delta_allowance=balance_delta
)

final_conf = collections.OrderedDict()

if not configurations:
    print("No matching configuration groups found.")
else:
    i = 0
    for p_anchor_key, group_data in configurations.items():
        final_conf[fc1_h_widths[i]] = {
            'anchor_params': p_anchor_key
        }
        print(f"\n--- Anchor (1-HL width {fc1_h_widths[i]}) Params Target: {p_anchor_key} ---")
        
        
        for num_hl, configs in group_data.items():
            print(f"  {num_hl} Hidden Layer(s):")
            
            # Closest P Config
            widths_closest = configs['closest_p']
            params_closest = _calculate_params(input_features, output_classes, widths_closest, include_bias)
            diff_closest = params_closest - p_anchor_key if params_closest != float('inf') else "N/A"
            balance_closest = calculate_balance_score(widths_closest)
            print(f"    Closest P:       Widths={widths_closest}, Params={params_closest}, Diff={diff_closest}, Balance (StdDev)={balance_closest:.2f}")

            # Balanced Close P Config
            widths_balanced = configs['balanced_close_p']
            params_balanced = _calculate_params(input_features, output_classes, widths_balanced, include_bias)
            diff_balanced = params_balanced - p_anchor_key if params_balanced != float('inf') else "N/A"
            balance_balanced = calculate_balance_score(widths_balanced)
            print(f"    Balanced Close P: Widths={widths_balanced}, Params={params_balanced}, Diff={diff_balanced}, Balance (StdDev)={balance_balanced:.2f}")
            
            final_conf[fc1_h_widths[i]][num_hl] = {
                'closest': {
                    'widths': widths_closest,
                    'num_params': params_closest,
                    'diff_parm_anchor': diff_closest
                },
                'balanced': {
                    'widths': widths_balanced,
                    'num_params': params_balanced,
                    'diff_parm_anchor': diff_balanced
                }
            }
        
        i += 1
            
# Save the OrderedDict to a file
with open('fc_width_depth_confs.pkl', 'wb') as f:
    pickle.dump(final_conf, f)