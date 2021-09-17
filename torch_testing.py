import os
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda

from EyeNeuralNetwork import EyeNeuralNetwork, EyeDataset

def train_loop(dataloader, model, loss_fn, optimizer):
    num_batches = len(dataloader)
    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        """
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        """
    train_loss /= num_batches
    return train_loss

def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    return test_loss

def main():
    # Selecting device
    device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # Loading data
    training_data = EyeDataset(
        "data/training_meta_data.csv", 
        "data/raw/",
        device
    )
    test_data_set = EyeDataset(
        "data/testing_meta_data.csv", 
        "data/raw/",
        device
    )
    train_dataloader = DataLoader(training_data, batch_size=512, shuffle=True)
    test_dataloader = DataLoader(test_data_set, batch_size=256, shuffle=True)


    model_load_path = 'model_weights_0.pth'
    model_save_path = 'model_weights_0.pth'

    # Initializing network
    model = EyeNeuralNetwork().to(device)
    learning_rate = 1e-2
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    epoch = 0
    epochs = 1101
    avg_loss_training_history = []
    avg_loss_testing_history = []

    # loading model
    if model_load_path is not None and os.path.exists(model_load_path):
        print("Loading existing model!")

        checkpoint = torch.load(model_load_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        avg_loss_training_history = checkpoint['avg_loss_training_history']
        avg_loss_testing_history = checkpoint['avg_loss_testing_history']

        model.train()

    # Training
    while epoch < epochs:
        train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loss = test_loop(test_dataloader, model, loss_fn)
        
        avg_loss_training_history.append(train_loss)
        avg_loss_testing_history.append(test_loss)
        
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss_training_history': avg_loss_training_history,
                'avg_loss_testing_history': avg_loss_testing_history,
                }, model_save_path)
        
        print(f" Epoch {epoch}: train_loss={train_loss:>8f}, test_loss={test_loss:>8f}", end='\n' if epoch % 20 == 0 or epoch == epochs else '\r')
        
        epoch += 1
    print("\n\nDone!")

    # Plotting results
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('avg loss training', color=color)
    ax1.plot(list(range(len(avg_loss_training_history))), avg_loss_training_history, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('avg loss testing', color=color)
    ax2.plot(list(range(len(avg_loss_testing_history))), avg_loss_testing_history, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()