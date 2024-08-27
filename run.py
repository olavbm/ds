from functools import partial
import os
import tempfile
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray import train
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self, h1=512, h2=512):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, h1),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(h2, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

config = {
    "h1": tune.choice([2 ** i for i in range(9)]),
    "h2": tune.choice([2 ** i for i in range(9)]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16])
}

learning_rate = 1e-3
epochs = 5
scheduler = ASHAScheduler(
    metric="loss",
    mode="min",
    max_t=5,
    grace_period=1,
    reduction_factor=2,
)

def run_agent(config, num_samples):

    # Initialize the loss function

    def train_loop(config):
        dataloader = DataLoader(training_data, batch_size=config["batch_size"])
        model = NeuralNetwork(config["h1"], config["h2"])
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"])
        size = len(dataloader.dataset)
        # Set the model to training mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * config["batch_size"] + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        train.report({"loss": loss.item() })



    def test_loop(dataloader, model, loss_fn):
        # Set the model to evaluation mode - important for batch normalization and dropout layers
        # Unnecessary in this situation but added for best practices
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        with torch.no_grad():
            for X, y in dataloader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



    result = tune.run(
        partial(train_loop),
        resources_per_trial={"cpu": 4},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        )
    best_trial = result.get_best_trial("loss", "min", "last")
    print(best_trial)
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print("Done!")

run_agent(config, num_samples = 20)
