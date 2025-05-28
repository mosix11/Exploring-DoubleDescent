import os
import json
import plotly.graph_objects as go

# Parameter range and experiment setup
param_range = [
    1, 4, 8, 12, 18, 20, 22, 24, 28, 32, 36, 38, 44, 56, 80, 96, 128,
    160, 192, 208, 216, 224, 232, 240, 248, 256, 264, 280, 296, 304,
    320, 336, 344, 368, 392, 432, 464, 512, 768, 1024, 2048, 3072,
    4096, 8192, 16384
]

experiments = [
    ("FC1_MoG(smpls100000+ftrs512+cls30+0.2Noise)_Parallel_Seed22", "Random Init"),
    ("FC1_MoG(smpls100000+ftrs512+cls30+0.2Noise)_WeightReuse_Seed22", "Weight Reuse"),
    ("FC1_MoG(smpls100000+ftrs512+cls30+0.2Noise)_WeightReuseFreeze_Seed22", "Weight Reuse + Freeze")
]

base_dir = "outputs/modelwise"
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Distinct colors for each experiment

def load_experiment_data():
    data = []
    for exp_dir, exp_name in experiments:
        exp_data = {
            "name": exp_name,
            "h_values": [],
            "best_train_loss": [],
            "final_train_loss": [],
            "best_train_acc": [],
            "final_train_acc": [],
            "best_test_loss": [],
            "final_test_loss": [],
            "best_test_acc": [],
            "final_test_acc": [],
        }
        
        full_path = os.path.join(base_dir, exp_dir)
        for h in param_range:
            prefix = f"fc1|h{h}_mog"
            subdirs = [d for d in os.listdir(full_path) if d.startswith(prefix)]
            if not subdirs:
                continue
            
            results_path = os.path.join(full_path, subdirs[0], "log", "results.json")
            if not os.path.exists(results_path):
                continue
                
            with open(results_path) as f:
                results = json.load(f)
                
            exp_data["h_values"].append(h)
            for metric in ["Train/Loss", "Train/ACC", "Test/Loss", "Test/ACC"]:
                for phase in ["best", "final"]:
                    key = f"{phase}_{metric.lower().replace('/', '_')}"
                    exp_data[key].append(results[phase][metric])
        
        data.append(exp_data)
    return data

def create_figures(data):
    metrics = [
        ("Train/Loss", "Loss", ["best_train_loss", "final_train_loss"]),
        ("Train/ACC", "Accuracy", ["best_train_acc", "final_train_acc"]),
        ("Test/Loss", "Loss", ["best_test_loss", "final_test_loss"]),
        ("Test/ACC", "Accuracy", ["best_test_acc", "final_test_acc"])
    ]
    
    figures = []
    for title, ylabel, keys in metrics:
        fig = go.Figure()
        for exp_idx, exp_data in enumerate(data):
            h_values = exp_data["h_values"]
            color = colors[exp_idx]
            
            # Best values
            fig.add_trace(go.Scatter(
                x=h_values,
                y=exp_data[keys[0]],
                mode="lines+markers",
                name=f"{exp_data['name']} (Best)",
                line=dict(color=color, dash="solid"),
                marker=dict(symbol="circle")
            ))
            
            # Final values
            fig.add_trace(go.Scatter(
                x=h_values,
                y=exp_data[keys[1]],
                mode="lines+markers",
                name=f"{exp_data['name']} (Final)",
                line=dict(color=color, dash="dot"),
                marker=dict(symbol="x")
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Number of Parameters (log scale)",
            yaxis_title=ylabel,
            xaxis_type="log",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            template="plotly_white",
            margin=dict(l=50, r=20, t=60, b=40)
        )
        figures.append(fig)
    return figures

# Generate and show figures
experiment_data = load_experiment_data()
figures = create_figures(experiment_data)

for fig in figures:
    fig.show()
