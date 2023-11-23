"""
Ésta es la versión 5 del modelo Transformer. En ésta versión se implementa un modelo de clasificación
idéntico al de la versión 4 pero usando los retornos logarítmicos en lugar de los precios.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from PricesManager import PricesManager

# Hyperparameters
time_ahead_to_predict = 1
# price_diff_threshold = 0.0001
batch_size = 32
block_size = 60
max_iters = 10000
eval_interval = 250
eval_iters = 250
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
# n_embed = 32 # 64 / 32 = 2
# n_head = 2
# n_layer = 2
dropout = 0.2
embeddings = 16
dense_layer = 64

#--------------------------#

def preprare_data(prices, block_size, time_ahead_to_predict):
    # Calculate the logaritmic returns
    returns = np.log(prices) - np.log(prices.shift(1))
    # Drop the NaN values
    returns = returns.dropna()
    # Scaler for the returns
    scaler = StandardScaler()   
    # Slice the data using pandas instead of a loop
    total_windows = pd.DataFrame()
    for i in range(0, block_size + time_ahead_to_predict):
        total_windows = pd.concat([total_windows, 
                                   returns.shift(-i)],
                                  ignore_index=True,
                                  axis=1)
    # total_windows = pd.concat([returns.shift(-i) for i in range(0,block_size + minutes_ahead_to_predict)], axis=1).dropna()
    # Drop the NaN values
    total_windows = total_windows.dropna()
    # Calculate the mean and standard deviation for each window
    # means = total_windows.mean(axis=1)
    # stds = total_windows.std(axis=1)

    # Normalize the sequences using the scaler
    total_windows = scaler.fit_transform(total_windows)
    pd.DataFrame(total_windows)
    # Separate the x and y values
    x_windows = total_windows[:, :-time_ahead_to_predict]
    y_values = total_windows[:, -1]

    # Initialize an array for the new y_labels
    # y_labels = np.zeros((y_values.shape[0], 3))

    # Make the labels a 3 column vector
    # y_labels[y_values > price_diff_threshold, 0] = 1  # Price goes up by x%
    # y_labels[(y_values <= price_diff_threshold) & (y_values >= -price_diff_threshold), 1] = 1  # Price stays the same +- x%
    # y_labels[y_values < -price_diff_threshold, 2] = 1  # Price goes down by x%

    # Normalize the sequences using vectorization
    # x_windows = (total_windows.iloc[:, :-(time_ahead_to_predict)].values - means.values[:, None]) / stds.values[:, None]

    # create torch tensors
    x_windows = torch.tensor(x_windows, dtype=torch.float32)
    y_labels = torch.tensor(y_values, dtype=torch.float32)

    #create test and train data
    train_size = int(0.8 * len(x_windows))
    test_size = len(x_windows) - train_size
    #divide the data into train and test based on the first 80% and the last 20%
    x_train, x_test = x_windows[:train_size], x_windows[train_size:]
    y_train, y_test = y_labels[:train_size], y_labels[train_size:]

    x_train = x_train[:len(x_train) - len(x_train) % block_size]
    y_train = y_train[:len(y_train) - len(y_train) % block_size]
    x_test = x_test[:len(x_test) - len(x_test) % block_size]
    y_test = y_test[:len(y_test) - len(y_test) % block_size]

    return x_train, y_train, x_test, y_test

def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    if split == 'train':
        x, y = x_train, y_train
    else:
        x, y = x_test, y_test
    ix = torch.randint(len(y) - block_size, (batch_size,))
    #torch.stack -> concatenates a sequence of tensors along a new dimension
    x = torch.stack([x[i] for i in ix]) 
    y = torch.stack([y[i] for i in ix])
    
    return x, y

# When we're not doing back propagation is a good practice to use this decorator
@torch.no_grad()
def estimate_loss():
    out = {}
    # Put it in evaluation mode (for dropout and batchnorm, usually)
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            pred, loss = model(x=X,targets=Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # we put it back to training mode
    return out

class Time2Vector(nn.Module):
    """One head of self-attention """
    def __init__(self, block_size, embeddings, dropout):
        super().__init__()
        self.linear = nn.Linear(block_size, block_size, bias=False)
        self.periodic = nn.Linear(block_size, block_size, bias=False)
        self.proj = nn.Linear(2, embeddings, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T = x.shape
        # B is the number of instances in the Barch; T is the number of prices (Tokens) we use as an input, defined by block_size
        time_linear=self.linear(x) # (B,T) 
        time_linear = time_linear.unsqueeze(-1) # (B,T,1)
        # Apply the periodic layer and then the sin function.
        time_periodic=torch.sin(self.periodic(x)) # (B,T)
        time_periodic = time_periodic.unsqueeze(-1) # (B,T,1)
        # Concatenate the two tensors in the last dimension
        out = torch.cat([time_linear, time_periodic], dim=-1) # (B,T,2)
        return out

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            num_layers,
                            batch_first=True,dropout=dropout)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        return x
    
class T2V_LSTM_Model(nn.Module):
    # This is a simpler model that uses a concatenation of
    # LSTM and Time2Vector as the input to the convolutional layers
    def __init__(self):
        # Initialize the parent class nn.Module
        super().__init__()
        self.t2v = Time2Vector(block_size, embeddings, dropout)
        self.LSTM = LSTM(input_size=block_size, hidden_size=1, num_layers=1)
        # Convolutions to get the desired output dimension
        self.conv1 = nn.Linear(block_size, 1)

    
    def forward(self, x, targets = None):
        B, T = x.shape # idx and targets is (B,T)
        x = self.t2v(x) # (B,T,embeddings)
        x = self.LSTM(x) # (B,T,1)
        x = x.squeeze(-1) # (B,T)
        # Convolution layer to get the desired output dimension
        x = self.conv1(x) # (B,T) @ (T,1) -> (B,1)

        # If targets is not None, we compute the loss
        if targets is None:
            loss = None
        else:
            B, T= x.shape
            # targets = targets.view(B) #targets.view(B*T)
            loss = F.mse_loss(x, targets)

        return x, loss


# Read the data
prices = pd.read_csv('Prices_Data/BTCUSDT.csv', 
                     index_col=0,
                     parse_dates=True,
                     encoding='utf-16',
                     sep=';')
prices = pd.DataFrame(prices)
# We just use the closing price
prices = pd.DataFrame(prices['Close'])
# prices = prices.resample('1H').last()

# Prepare train and test data    
x_train, y_train, x_test, y_test = preprare_data(prices, block_size, time_ahead_to_predict)

""" Parameters to test
# minutes_ahead_to_predict_list = [5, 10, 15, 30, 60, 120, 240, 480, 960, 1440]
# price_diff_threshold_list = [0.02, 0.01, 0.005, 0.0025, 0.001]
# block_size_list = [10, 15, 30, 60, 120, 240, 480, 960, 1440]
# n_head_list = [1, 2, 4]
# embeddings_list = [8, 16, 32, 64]
# dense_layer_list = [16, 32, 64]
# max_iters_list = 2000
# eval_interval_list = 250
# eval_iters_list = 250
# learning_rate_list = 1e-3

# for min_ahead in minutes_ahead_to_predict_list:
#     #let only the values of block_size_list that are smaller than minutes_ahead_to_predict
#     block_size_list = [x for x in block_size_list if x >= min_ahead]
#     for blk_size in block_size_list:
#         for n_h in n_head_list:
#             for emb in embeddings_list:
#                 for price_diff in price_diff_threshold_list:
#                     for d_layer in dense_layer_list:
#                         print('--------------------------------------------------')
#                         print(f'Parameters: block_size = {blk_size}, n_head = {n_h}, embeddings = {emb}, dense_layer = {d_layer}')
#                         # Get the data with the parameters of this iteration
#                         x_train, y_train, x_test, y_test = preprare_data(prices, blk_size, min_ahead, price_diff)

#                         model = Model(block_size=blk_size, n_head=n_h, embeddings=emb, dense_layer=d_layer)
#                         m = model.to(device)

#                         # create a PyTorch optimizer
#                         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#                         for iter in range(max_iters):
#                             #every once in a while evaluate the loss on the train and validation set
#                             if iter % eval_interval == 0 or iter== max_iters-1:
#                                 losses = estimate_loss()
#                                 print(f'Step iter {iter}: train loss = {losses["train"]:.4f}, val loss = {losses["val"]:.4f}')

#                             # sample a batch of data
#                             xb, yb = get_batch('train')
                            
#                             # evaluate the loss
#                             pred, loss = model(xb, yb)
#                             optimizer.zero_grad(set_to_none=True)
#                             loss.backward()
#                             optimizer.step()
#                         print('--------------------------------------------------')
"""

model = T2V_LSTM_Model()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# m, optimizer = ipex.optimize(model, optimizer, dtype=torch.float32)

for iter in range(max_iters):
    #every once in a while evaluate the loss on the train and validation set
    if iter % eval_interval == 0 or iter== max_iters-1:
        losses = estimate_loss()
        print(f'Step iter {iter}: train loss = {losses["train"]:.4f}, val loss = {losses["val"]:.4f}')

    # sample a batch of data
    xb, yb = get_batch('train')
    
    # evaluate the loss
    pred, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# # We'll transform the predictions back to the original scale
# means = torch.tensor(means, dtype=torch.float32)
# stds = torch.tensor(stds, dtype=torch.float32)

# means_train, means_test = means[:train_size], means[train_size:]
# stds_train, stds_test = stds[:train_size], stds[train_size:]

# means_test = means_test.to(device)
# means_train = means_train.to(device)
# stds_test = stds_test.to(device)
# stds_train = stds_train.to(device)

# # Run the model on the test and train data. Discard the loss.
# y_pred_test, _ = model(x_test)
# y_pred_train, _ = model(x_train)

# # Convert the predictions back to the original scale
# prices_pred_test = y_pred_test * stds_test + means_test
# prices_pred_train = y_pred_train * stds_train + means_train

# prices_labels_test = y_test * stds_test + means_test
# prices_labels_train = y_train * stds_train + means_train

# #convert to numpy arrays
# prices_pred_test = prices_pred_test.cpu().detach().numpy()
# prices_pred_train = prices_pred_train.cpu().detach().numpy()

# #Calculate the absolute error of each prediction
# errors_test = prices_pred_test - prices_labels_test
# errors_train = prices_pred_train - prices_labels_train

# #plot the predictions
# plt.figure(figsize=(12,6))
# plt.plot(prices_pred_test, label='predictions')
# plt.plot(prices_labels_test, label='labels')
# plt.legend()
