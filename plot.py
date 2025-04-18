import os
import pickle
import re
import matplotlib.pyplot as plt
from pathlib import Path


h_values = []
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# results_dir = Path('outputs/results/fc1_mnist')
results_dir = Path('outputs/results/fc1_cifar10')

for filename in os.listdir(results_dir):
    match = re.match(r"res_param(\d+)\.pkl", filename)
    if match:
        h = int(match.group(1))
        with open(results_dir / filename, "rb") as f:
            data = pickle.load(f)
        h_values.append(h)
        train_losses.append(data["train_loss"])
        test_losses.append(data["test_loss"])
        train_accuracies.append(data["train_acc"])
        test_accuracies.append(data["test_acc"])

sorted_data = sorted(zip(h_values, train_losses, test_losses, train_accuracies, test_accuracies))
h_values, train_losses, test_losses, train_accuracies, test_accuracies = zip(*sorted_data)

fig, axs = plt.subplots(2, 1, figsize=(8, 10))

axs[0].plot(h_values, train_losses, 'o-r', label='train')
axs[0].plot(h_values, test_losses, 'o-b', label='test')
axs[0].set_xlabel("Hidden layer size (h)")
axs[0].set_ylabel("loss")
axs[0].set_xscale('log') 
axs[0].legend()
axs[0].grid(True)

axs[1].plot(h_values, train_accuracies, 'o-r', label='train')
axs[1].plot(h_values, test_accuracies, 'o-b', label='test')
axs[1].set_xlabel("Hidden layer size (h)")
axs[1].set_ylabel("accuracy")
axs[1].set_xscale('log') 
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
