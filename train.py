from src.datasets import MNIST, CIFAR10, FashionMNIST

ds = MNIST()
dl = ds.get_train_dataloader()
for batch in dl:
    print(len(batch))
    
    