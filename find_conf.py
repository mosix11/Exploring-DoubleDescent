import collections
import math

# --- Helper Functions ---

def _calculate_params(input_dim: int, output_dim: int, hidden_widths: list[int], count_bias: bool) -> int | float:
    """
    Calculates the number of parameters for a given FCNN architecture.
    Returns float('inf') if a dimension is invalid (e.g., <= 0),
    which can happen if a calculated width becomes non-positive.
    """
    params = 0
    all_dims = [input_dim] + hidden_widths + [output_dim]
    
    for i in range(len(all_dims) - 1):
        dim_in = all_dims[i]
        dim_out = all_dims[i+1]
        
        # Basic check for valid dimensions during calculation
        if dim_in <= 0 or dim_out <= 0:
            return float('inf') 

        params += dim_in * dim_out  # Weights
        if count_bias:
            params += dim_out  # Biases
    return params

def _solve_for_last_hidden_width(
    p_anchor: int, 
    input_dim: int, 
    output_dim: int, 
    count_bias: bool,
    prefix_hidden_widths: list[int] # Widths [w1, ..., w_{L-1}] for an L-HL network
) -> int | None:
    """
    Solves for the optimal integer width of the L-th (last) hidden layer (w_L),
    given the L-1 prefix_hidden_widths, to make the L-HL network's total parameters
    closest to p_anchor. Returns the best w_L (>=1) or None.
    """
    params_prefix_layers = 0
    
    # Determine width of the (L-1)th hidden layer, or input_dim if L=1
    if not prefix_hidden_widths: # Solving for w1 in a 1-HL network
        width_of_L_minus_1_layer = input_dim
    else: # Solving for wL where L >= 2
        # Calculate parameters for layers: Input -> w1 -> ... -> w_{L-1}
        current_dims_for_prefix = [input_dim] + prefix_hidden_widths
        for i in range(len(current_dims_for_prefix) - 1):
            d_in = current_dims_for_prefix[i]
            d_out = current_dims_for_prefix[i+1] # d_out is w_i in prefix
            params_prefix_layers += d_in * d_out
            if count_bias:
                params_prefix_layers += d_out
        width_of_L_minus_1_layer = prefix_hidden_widths[-1]

    # Equation for w_L (last hidden layer):
    # P_anchor ~ params_prefix_layers + (width_of_L_minus_1_layer + B) * w_L + (w_L + B) * output_dim
    # (where B=1 if count_bias else 0)
    # Rewriting: P_anchor - params_prefix_layers ~ w_L * (width_of_L_minus_1_layer + B + output_dim) + B * output_dim
    
    numerator = p_anchor - params_prefix_layers
    denominator = width_of_L_minus_1_layer + output_dim

    if count_bias:
        numerator -= output_dim  # Account for the (B * output_dim) term from output layer's bias
        denominator += 1         # Account for B in (width_of_L_minus_1_layer + B + output_dim)

    if denominator <= 0: # Avoid division by zero or nonsensical denominator
        return None

    wL_float = numerator / denominator

    wL_candidates = []
    floor_wL = math.floor(wL_float)
    ceil_wL = math.ceil(wL_float)

    if floor_wL >= 1:
        wL_candidates.append(floor_wL)
    if ceil_wL >= 1 and ceil_wL != floor_wL: # Add if different and >= 1
        wL_candidates.append(ceil_wL)
    
    if not wL_candidates: # No valid w_L (>=1) found
        return None

    best_wL_for_this_prefix = None
    min_diff_for_this_prefix = float('inf')

    for wL_cand in wL_candidates:
        # Construct the full list of hidden widths for this L-HL network
        current_total_hidden_widths = prefix_hidden_widths + [wL_cand]
        actual_total_params = _calculate_params(input_dim, output_dim, current_total_hidden_widths, count_bias)
        
        if actual_total_params == float('inf'): # Skip if calc params failed
            continue

        diff = abs(actual_total_params - p_anchor)

        if diff < min_diff_for_this_prefix:
            min_diff_for_this_prefix = diff
            best_wL_for_this_prefix = wL_cand
        elif diff == min_diff_for_this_prefix:
            # Tie-breaking: prefer smaller w_L if differences are identical
            if best_wL_for_this_prefix is None or wL_cand < best_wL_for_this_prefix:
                best_wL_for_this_prefix = wL_cand
                
    return best_wL_for_this_prefix

def _find_best_config_for_L_hidden_layers(
    num_total_hidden_layers: int, 
    p_anchor: int, 
    k_anchor_1hl_width: int, # Max width for iterating w1, w2, ...
    input_dim: int, 
    output_dim: int, 
    count_bias: bool
) -> list[int] | None:
    """
    Finds the configuration [w1, ..., wL] for an L-hidden-layer network
    that results in a total parameter count closest to p_anchor.
    Constraints: 1 <= wi <= k_anchor_1hl_width for i=1 to L-1. wL is solved.
    """
    L = num_total_hidden_layers
    best_overall_widths = None
    min_overall_diff = float('inf')

    # Tie-breaking helper for selecting the best_overall_widths across all prefixes
    def _update_best_overall_config(new_widths_candidate, p_actual_candidate):
        nonlocal best_overall_widths, min_overall_diff
        if new_widths_candidate is None or p_actual_candidate == float('inf'):
            return

        current_diff = abs(p_actual_candidate - p_anchor)

        if current_diff < min_overall_diff:
            min_overall_diff = current_diff
            best_overall_widths = new_widths_candidate
        elif current_diff == min_overall_diff:
            if best_overall_widths is None: # Should be initialized if min_overall_diff isn't inf
                 best_overall_widths = new_widths_candidate
            # Prefer smaller sum of widths, then lexicographically smaller widths
            elif sum(new_widths_candidate) < sum(best_overall_widths):
                best_overall_widths = new_widths_candidate
            elif sum(new_widths_candidate) == sum(best_overall_widths) and new_widths_candidate < best_overall_widths:
                best_overall_widths = new_widths_candidate
    
    if L == 0: # Linear model
        actual_params = _calculate_params(input_dim, output_dim, [], count_bias)
        _update_best_overall_config([], actual_params)
        return best_overall_widths # Will be [] if actual_params is not inf

    elif L == 1: # One hidden layer
        w1_solved = _solve_for_last_hidden_width(p_anchor, input_dim, output_dim, count_bias, [])
        if w1_solved is not None:
            current_widths = [w1_solved]
            actual_params = _calculate_params(input_dim, output_dim, current_widths, count_bias)
            _update_best_overall_config(current_widths, actual_params)

    elif L == 2: # Two hidden layers [w1, w2]
        for w1 in range(1, k_anchor_1hl_width + 1):
            w2_solved = _solve_for_last_hidden_width(p_anchor, input_dim, output_dim, count_bias, [w1])
            if w2_solved is not None:
                current_widths = [w1, w2_solved]
                actual_params = _calculate_params(input_dim, output_dim, current_widths, count_bias)
                _update_best_overall_config(current_widths, actual_params)

    elif L == 3: # Three hidden layers [w1, w2, w3]
        for w1 in range(1, k_anchor_1hl_width + 1):
            for w2 in range(1, k_anchor_1hl_width + 1): # w2 also constrained by k_anchor_1hl_width
                w3_solved = _solve_for_last_hidden_width(p_anchor, input_dim, output_dim, count_bias, [w1, w2])
                if w3_solved is not None:
                    current_widths = [w1, w2, w3_solved]
                    actual_params = _calculate_params(input_dim, output_dim, current_widths, count_bias)
                    _update_best_overall_config(current_widths, actual_params)
    else:
        # This version currently supports up to L=3 hidden layers explicitly.
        # For L > 3, a recursive approach for iterating w1..w_{L-1} would be needed.
        print(f"Warning: Configuration search for {L} hidden layers is not implemented beyond L=3.")
        pass 

    return best_overall_widths

# --- Main Function ---

def get_network_configurations(
    input_dim: int, 
    output_dim: int, 
    target_hidden_layers: list[int], 
    count_bias: bool, 
    max_anchor_1hl_width: int
) -> collections.OrderedDict:
    """
    Finds FCNN configurations anchored on 1-hidden-layer networks.

    For each width `k` (from 1 to `max_anchor_1hl_width`) of a 1-hidden-layer network,
    its parameter count `P_anchor` is calculated. Then, for each number of hidden
    layers `L` specified in `target_hidden_layers` (where L can also be 1),
    the function finds an L-hidden-layer configuration whose total parameters
    are closest to `P_anchor`.

    Args:
        input_dim: Number of input features.
        output_dim: Number of output units.
        target_hidden_layers: A list of integers for the number of hidden layers
                              to find configurations for (e.g., [1, 2, 3]).
                              It's assumed that 1-HL networks are the anchor.
        count_bias: Boolean, whether to include bias parameters.
        max_anchor_1hl_width: The maximum width to iterate for the single hidden
                               layer of the anchor (1-HL) network.

    Returns:
        An collections.OrderedDict where:
            - Keys are the parameter counts of the 1-hidden-layer anchor networks.
            - Values are OrderedDicts, mapping the number of hidden layers (L)
              to the list of determined hidden layer widths for that L.
              If a configuration for a specific L could not be found to match
              an anchor, that L might be missing from the inner dict.
    """
    if not isinstance(input_dim, int) or input_dim < 1:
        raise ValueError("input_dim must be a positive integer.")
    if not isinstance(output_dim, int) or output_dim < 1:
        raise ValueError("output_dim must be a positive integer.")
    if not isinstance(target_hidden_layers, list) or not target_hidden_layers:
        raise ValueError("target_hidden_layers must be a non-empty list.")
    if not all(isinstance(hl, int) and hl >= 0 for hl in target_hidden_layers): # 0 for linear
        raise ValueError("All elements in target_hidden_layers must be non-negative integers.")
    if not isinstance(count_bias, bool):
        raise ValueError("count_bias must be a boolean.")
    if not isinstance(max_anchor_1hl_width, int) or max_anchor_1hl_width < 1:
        raise ValueError("max_anchor_1hl_width must be a positive integer.")

    result_dict = collections.OrderedDict()
    
    # Ensure target_hidden_layers are sorted for consistent output order within groups
    sorted_target_hl = sorted(list(set(target_hidden_layers)))

    for k_1hl in range(1, max_anchor_1hl_width + 1):
        anchor_1hl_widths = [k_1hl]
        p_anchor = _calculate_params(input_dim, output_dim, anchor_1hl_widths, count_bias)

        if p_anchor == float('inf'): # Should not happen if k_1hl >= 1
            continue

        current_group_configs = collections.OrderedDict()
        all_depths_found_for_this_anchor = True

        for L_target in sorted_target_hl:
            if L_target == 1: # The anchor network itself
                # Check if P_anchor derived from [k_1hl] is what we want to store.
                # We are defining the anchor by [k_1hl], so its config is fixed.
                # We could also re-solve it using _find_best_config to see if it finds k_1hl,
                # but that's redundant if p_anchor is already derived from it.
                config_for_L = anchor_1hl_widths
            else:
                config_for_L = _find_best_config_for_L_hidden_layers(
                    L_target, p_anchor, k_1hl, # k_1hl is the k_anchor_width constraint
                    input_dim, output_dim, count_bias
                )
            
            if config_for_L is not None:
                current_group_configs[L_target] = config_for_L
            else:
                # If we couldn't find a config for one of the target depths,
                # this anchor point (p_anchor) does not yield a complete group.
                all_depths_found_for_this_anchor = False
                break 
        
        if all_depths_found_for_this_anchor and len(current_group_configs) == len(sorted_target_hl):
            if p_anchor not in result_dict: # Avoid overwriting if different k_1hl lead to same P_anchor
                result_dict[p_anchor] = current_group_configs
            # else:
            # Potentially, multiple k_1hl could lead to the same p_anchor.
            # The current logic takes the first k_1hl that generates this p_anchor.
            # Or, one might want to store all such k_1hl values, but that complicates the output.

    return result_dict



# --- Example Usage ---

# Define the parameters for the search
input_features = 512
output_classes = 30
layers_to_test = [1, 2, 3]  # Number of hidden layers for configurations
include_bias = True
max_k_for_1hl_anchor = 512   # Max width for the 1-hidden-layer anchor network

print(f"Searching configurations for:")
print(f"  Input Dimensions: {input_features}")
print(f"  Output Dimensions: {output_classes}")
print(f"  Target Hidden Layers: {layers_to_test}")
print(f"  Count Bias: {include_bias}")
print(f"  Max Anchor (1-HL) Width: {max_k_for_1hl_anchor}\n")

# Call the function
configurations = get_network_configurations(
    input_dim=input_features,
    output_dim=output_classes,
    target_hidden_layers=layers_to_test,
    count_bias=include_bias,
    max_anchor_1hl_width=max_k_for_1hl_anchor
)

# Print the results
if not configurations:
    print("No matching configuration groups found with the given parameters.")
else:
    print(f"Found {len(configurations)} groups of configurations (keyed by 1-HL anchor's parameter count):")
    i = 1
    for p_anchor_key, group_configs in configurations.items():
        print(f"\n--- Anchor (1-HL width={i}) Params Target: {p_anchor_key} ---")
        i+=1
        for num_hl, widths in group_configs.items():
            actual_params = _calculate_params(input_features, output_classes, widths, include_bias)
            param_diff = actual_params - p_anchor_key if actual_params != float('inf') else "N/A"
            print(f"  {num_hl} Hidden Layer(s): Widths={widths if widths is not None else 'N/A'}, "
                  f"Actual Params={actual_params if actual_params != float('inf') else 'Error'}, "
                  f"Diff from Anchor Target={param_diff}")