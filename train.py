from src.datasets import MNIST, CIFAR10, FashionMNIST

import matplotlib.pyplot as plt

ds = MNIST(flatten=False)
dl = ds.get_train_dataloader()
for batch in dl:
    print(batch[0].shape)
    print(batch[1][211])
    # plt.imshow(batch[0][0].reshape((28,28)).detach().cpu(),)
    # plt.axis('off')  # Optional: Turn off axis numbers and ticks
    # plt.show()
    break
    
    