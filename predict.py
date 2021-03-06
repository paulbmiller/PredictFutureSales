"""
The goal is to predict total sales for every product and store for the next
month.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from statsmodels.tools.eval_measures import rmse
from tqdm import tqdm
import time
import math

# Some code was taken from the notebook
# https://www.kaggle.com/karanjakhar/simple-and-easy-aprroach-using-lstm


class LSTM(nn.Module):
    """Class for the LSTM model"""
    def __init__(self, device, hidden_layer=96, num_layers=4,
                 output_size=1, dropout=0, val=False):
        super().__init__()
        if val:
            self.input_size = 32
        else:
            self.input_size = 33
        self.hidden_layer = hidden_layer
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(self.input_size, hidden_layer,
                            num_layers=num_layers, dropout=dropout)
        
        self.linear = nn.Linear(hidden_layer, output_size)
        
        self.hidden_cell = (
            torch.zeros(num_layers, 1, self.hidden_layer).to(device),
            torch.zeros(num_layers, 1, self.hidden_layer).to(device))
    
    def forward(self, x):
        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        predictions = self.linear(lstm_out)
        return predictions


def train(epochs, training_set, batch_size, model, loss_fn, optimizer, device,
          lr, max_epochs=None, curr_epoch=None):
    if curr_epoch is None:
        curr_epoch = 0
    if max_epochs is None:
        max_epochs = epochs
    
    data_loader = torch.utils.data.DataLoader(training_set, batch_size,
                                              shuffle=False, num_workers=0)
    model.train()
    num_batches = len(data_loader)
    losses = np.array([])
    for i in range(epochs):
        bar_format = \
            'Training ep. {}/{} : '.format(
                curr_epoch + i + 1, max_epochs) + '{l_bar}{bar}{r_bar}'
        running_loss = 0.0
        for current_batch in tqdm(data_loader, unit='batches',
                                  total=num_batches, bar_format=bar_format):
            seq = current_batch[:, :-1].reshape(-1, 1, model.input_size)
            label = current_batch[:, -1:].reshape(-1, 1, 1)
            
            optimizer.zero_grad()
            model.hidden_cell = (
                torch.zeros(
                    model.num_layers, 1, model.hidden_layer).to(device),
                torch.zeros(
                    model.num_layers, 1, model.hidden_layer).to(device))
            y_pred = model(seq)

            loss = loss_fn(y_pred, label)
            loss.backward()
            optimizer.step()
    
    return losses


def evaluate(epochs, batch_size, model, loss_fn, optimizer, X_test, device, lr,
             y_test=None):
    model.eval()
    preds = torch.Tensor([]).to(device)
    
    X_test = X_test.to(device)
    
    X_test_dl = torch.utils.data.DataLoader(X_test,
                                            batch_size,
                                            shuffle=False,
                                            num_workers=0)
        
    for seq in tqdm(X_test_dl, total=len(X_test_dl),
                    bar_format='Testing : {l_bar}{bar}{r_bar}'):
        with torch.no_grad():
            model.hidden_cell = (
                torch.zeros(
                    model.num_layers, 1, model.hidden_layer).to(device),
                torch.zeros(
                    model.num_layers, 1, model.hidden_layer).to(device))
            pred = model(seq.reshape(-1, 1, model.input_size))
            
            preds = torch.cat((preds, pred))
    
    preds = preds.flatten()
    preds = torch.clamp(preds, 0, 20)
    
    if y_test is not None:
        y_test = y_test.to(device)
        preds = loss_fn(preds, y_test).item()
    
    return (model, type(optimizer), epochs, batch_size, lr, preds)


def RMSELoss(y_pred, label):
    criterion = nn.MSELoss()
    return torch.sqrt(criterion(y_pred, label))


def plot_12(results, res_num):
    n_rows, n_cols = 3, 4
    fig, axs = plt.subplots(n_rows, n_cols)
    fig.suptitle('Results {}'.format(res_num))
    
    for i in range(n_rows * n_cols):
        row = i // n_cols
        col = i % n_rows
        axs[row, col].set_title(
                'Result {}'.format(i + res_num * 12))
        axs[row, col].plot(results[i][5])


def grid_search(models, epoch_list, optimizers, batch_sizes, train_val_set,
                test_val_set, lrs, y_val):
    num_models = len(models) * len(lrs) * len(optimizers) * len(batch_sizes)
    i = 1
    train_losses = []
    val_info = []
    epoch_list.sort()
    max_epochs = epoch_list[-1]
    for mod in models:
        model = mod.to(device)
        
        for lr in lrs:
            for optim in optimizers:
                optimizer = optim(model.parameters(), lr=lr)
                for batch_size in batch_sizes:
                    # tqdm.write('\nGrid search : {} out of {}\n'.format(
                    #     i, num_models))
                    for _, module in model.named_children():
                        module.reset_parameters()
                        
                    i += 1
                    
                    epochs_trained = 0
                        
                    data_loader = torch.utils.data.DataLoader(train_val_set,
                                                              batch_size,
                                                              shuffle=False,
                                                              num_workers=0)
                    
                    for epochs in epoch_list:
                        train_losses.append(
                            train(epochs - epochs_trained, train_val_set,
                                  batch_size, model, loss_fn, optimizer,
                                  device, lr, max_epochs, epochs_trained))

                        epochs_trained += epochs - epochs_trained
    
                        val_info.append(
                            evaluate(epochs, batch_size, model, loss_fn,
                                 optimizer, test_val_set, device, lr,
                                 y_val))

    return train_losses, val_info


if __name__ == '__main__':
    
    
    sales_train = pd.read_csv('data/sales_train.csv')
    sales_train['date'] = pd.to_datetime(sales_train['date'],
                                         format='%d.%m.%Y')
    sales_train.drop('item_price', axis=1, inplace=True)
    sales_train.drop('date', axis=1, inplace=True)
    
    # Since the shop 0 is the same as shop 57 and 1 is shop 58
    sales_train['shop_id'] = sales_train['shop_id'].replace(57, 0)
    sales_train['shop_id'] = sales_train['shop_id'].replace(58, 1)
    
    sales_train = sales_train.groupby(['date_block_num', 'shop_id',
                                       'item_id'], as_index=False).sum()
    sales_train = sales_train.rename(columns={'item_cnt_day':'item_cnt_month'})
    
    sales_train = sales_train.pivot_table(index=['shop_id', 'item_id'],
                                          values=['item_cnt_month'],
                                          columns=['date_block_num'],
                                          fill_value=0, aggfunc='sum')
    
    test = pd.read_csv('data/test.csv', index_col='ID')
    
    # Take the shop/item combinations from the test set
    dataset = pd.merge(test, sales_train, on=['item_id', 'shop_id'],
                       how='left')
    dataset.fillna(0, inplace=True)
    dataset.drop(['shop_id', 'item_id'], axis=1, inplace=True)
    dataset = dataset.clip(0, 20)
    
    loss_fn = RMSELoss
    
    # Grid search
    """
    Otherwise I had a CUDA Error : Unspecified launch failure which may be
    driver related (increases training time by a factor of 5-6 on my machine
    but doesn't increase testing time).
    """

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_val_set = torch.from_numpy(dataset.values[:, :-1]).float().to(device)
    test_val_set = torch.from_numpy(dataset.values[:, 1:-1]).float().to(device)
    y_val = torch.from_numpy(dataset.values[:, -1]).float().to(device)
    """
    val = True
    
    epoch_list = [5]
    models = [
        LSTM(device, hidden_layer=192, num_layers=3, dropout=0.0, val=val)]
    
    optimizers = [torch.optim.Adam]
    
    batch_sizes = [256, 512, 1024]
    
    lrs = [1.5e-3, 2e-3, 2.5e-3, 3e-3]
    
    train_losses, val_info = \
        grid_search(models, epoch_list, optimizers, batch_sizes, train_val_set,
                    test_val_set, lrs, y_val)
    
    val_info.sort(key=lambda tup: tup[5])
    
    df = pd.DataFrame(data=val_info)
    df.columns = ['model', 'optimizer', 'epochs', 'batch_size', 'lr', 'RSME']
    df.to_csv('grid_search_results.csv', mode='a')
    """
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    training_set = torch.from_numpy(dataset.values).float().to(device)
    # for test we keep all the columns except the first one
    X_test = torch.from_numpy(dataset.values[:, 1:]).float().to(device)
    
    model = LSTM(device, hidden_layer=192, num_layers=3).to(
        device)
    epochs = 5
    optimizer = torch.optim.Adam
    batch_size = 1024
    lr = 1.5e-3
    optim = optimizer(model.parameters(), lr=lr)
    
    train(epochs, training_set, batch_size, model, loss_fn, optim, device,
          lr)
    
    eval_res = evaluate(epochs, batch_size, model, loss_fn, optim, X_test,
                        device, lr)
    
    preds = eval_res[5].cpu().numpy()
    
    df = pd.DataFrame(data={'ID': test.index, 'item_cnt_month': preds})

    df.to_csv('submissions/sub_gs3.csv', index=False)
    
    
