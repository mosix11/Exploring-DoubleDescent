from src.datasets import MNIST, CIFAR10, FashionMNIST
from src.models import FC1, FiveCNN
from src.trainers import Trainer
import matplotlib.pyplot as plt
from src.utils import nn_utils
import torch
from functools import partial
from pathlib import Path
import pickle

def start_training(results_dir: Path, checkpoints_dir: Path):
    max_epochs = 6000
    param_range = [
        3,
        4,
        7,
        9,
        10,
        20,
        30,
        40,
        45,
        47,
        49,
        50,
        51,
        53,
        55,
        60,
        70,
        80,
        90,
        100,
        110,
        128,
        150,
        170,
        200,
        256,
        512,
        1024,
    ]

    optim_cgf = {"type": "sgd", "lr": 1e-2, "momentum": 0.95}
    lr_schedule_cfg = {
        "type": "step_lr",
        "milestones": list(range(500, 6000, 500)),
        "gamma": 0.9,
    }
    subsample_size = 4000
    dataset = MNIST(
        batch_size=256,
        subsample_size=subsample_size,
        num_workers=8,
        valset_ratio=0.0,
        normalize_imgs=False,
        flatten=True,
    )

    loss_fn = torch.nn.MSELoss()

    def accuracy_metric(preds: torch.Tensor, targets: torch.Tensor) -> float:
        _, predicted = preds.max(1)
        num_samples = targets.size(0)
        correct = predicted.eq(targets.argmax(1)).sum().item()
        return correct / num_samples

    for idx, param in enumerate(param_range):

        if idx == 0:
            weight_init_method = nn_utils.init_xavier_uniform
        elif param > 50:
            weight_init_method = partial(nn_utils.init_normal, mean=0.0, std=0.1)
        else:
            weight_init_method = partial(nn_utils.init_normal, mean=0.0, std=0.1)

        
        model = FC1(
            input_dim=784,
            hidden_dim=param,
            ouput_dim=10,
            weight_init=weight_init_method,
            loss_fn=loss_fn,
            metric=accuracy_metric,
        )

        # In underparameterized regime, we reuse weights from the last trained model.
        if idx > 0 and param <= 50:
            last_trained_param = param_range[idx - 1]
            old_state_path = checkpoints_dir / Path(f"ckp_h{last_trained_param}.pth")
            old_state = torch.load(old_state_path)
            model.reuse_weights(old_state)

        early_stopping = False if param > 50 else True
        trainer = Trainer(
            max_epochs=max_epochs,
            optimizer_cfg=optim_cgf,
            lr_schedule_cfg=lr_schedule_cfg,
            early_stopping=early_stopping,
            run_on_gpu=True,
            use_amp=False,
            log_tensorboard=True,
            log_name=f"param{param}"
        )

        results = trainer.fit(model, dataset, resume=False)
        print(
            f"\n\n\nTraining the model with hidden layer size {param} finished with test loss of {results['test_loss']}, and test accuracy of {results['test_acc']}.\n\n\n"
        )

        torch.save(model.state_dict(), checkpoints_dir / Path(f"ckp_h{param}.pth"))
        result_path = results_dir / Path(f"res_param{param}.pkl")
        with open(result_path, 'wb') as f:
            pickle.dump(results, f)
        


# for batch in dl:
#     print(batch[0].shape)
#     print(batch[1][211])
#     # plt.imshow(batch[0][0].reshape((28,28)).detach().cpu(),)
#     # plt.axis('off')  # Optional: Turn off axis numbers and ticks
#     # plt.show()
#     break


if __name__ == "__main__":

    outputs_dir = Path("outputs")
    results_dir = outputs_dir / "results/FC1_mnist"
    checkpoints_dir = outputs_dir / "checkpoints/FC1_mnist"
    results_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir.mkdir(exist_ok=True, parents=True)

    start_training(results_dir, checkpoints_dir)
