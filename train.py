import os
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.FaceNeuralNetwork import FaceNeuralNetwork, FaceDataset


def train_loop(dataloader, model, loss_fn, optimizer):
    num_batches = len(dataloader)
    train_loss = 0

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(torch.clamp(pred, min=0.0, max=1.0), y)
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
            test_loss += loss_fn(torch.clamp(pred, min=0.0, max=1.0), y).item()

    test_loss /= num_batches
    return test_loss

def main():
    device = 'cpu' # 'cuda' if torch.cuda.is_available() else 'cpu'
    model_load_path = 'models/model_weights_p_B.pth'
    model_save_path = 'models/model_weights_p_B.pth'
    data_folder_name = 'data_p_B'
    learning_rate = 1e-2
    # loss_fn = nn.L1Loss()
    loss_fn = nn.MSELoss()
    max_epochs = 5801

    print('Using {} device'.format(device))

    # region Loading data
    print("Loading training data set")
    train_data = FaceDataset(
        data_folder_name + "/meta_data.csv", 
        data_folder_name + "/raw/",
        device,
        testing=False
    )
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True)
    
    print("Loading testing data set")
    test_data_set = FaceDataset(
        data_folder_name + "/meta_data.csv", 
        data_folder_name + "/raw/",
        device,
        testing=True
    )
    test_dataloader = DataLoader(test_data_set, batch_size=1, shuffle=True)
    # endregion

    # region Initialize model
    model = FaceNeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    epoch = 0
    avg_loss_training_history = []
    avg_loss_testing_history = []

    # loading model
    print("{} (Loading file) -> {} (Saving file)".format(model_load_path, model_save_path))
    if model_load_path is not None and os.path.exists(model_load_path):
        print("Loading existing model!")

        checkpoint = torch.load(model_load_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        avg_loss_training_history = checkpoint['avg_loss_training_history']
        avg_loss_testing_history = checkpoint['avg_loss_testing_history']

        model.train()
    # endregion

    # region Training
    print("Starting training")
    while epoch < max_epochs:
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
        
        print(f" Epoch {epoch}: train_loss={train_loss:>8f}, test_loss={test_loss:>8f}", end='\n' if epoch % 10 == 0 or epoch == max_epochs else '\r')
        
        epoch += 1
    print("\n\nDone!")
    # endregion

    # region Plotting results

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('avg loss train/test')
    ax1.plot(list(range(len(avg_loss_training_history))), avg_loss_training_history, color='tab:red')
    ax1.plot(list(range(len(avg_loss_testing_history))), avg_loss_testing_history, color='tab:blue')

    fig.tight_layout()
    plt.show()
    # endregion

if __name__ == "__main__":
    main()