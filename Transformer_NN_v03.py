import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 64
block_size = 50
max_iters = 1000
eval_interval = 100
eval_iters = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
# n_embed = 32 # 64 / 32 = 2
n_head = 4
n_layer = 2
dropout = 0.2
embeddings = 16
dense_layer = 256

#-----------
# torch.manual_seed(1337)

# Read the data
prices = pd.read_csv('Prices_Data/EURUSD.csv', index_col=0, parse_dates=True, encoding='utf-16', sep=';')

prices=prices.loc[:,'Close']

# Slice the data using pandas instead of a loop
total_windows = pd.concat([prices.shift(-i) for i in range(0,block_size+1)], axis=1).dropna()

# Calculate the mean and standard deviation for each window
means = total_windows.mean(axis=1)
stds = total_windows.std(axis=1)

# Normalize the sequences using vectorization
x_windows = (total_windows.iloc[:, :-1].values - means.values[:, None]) / stds.values[:, None]
y_labels = (total_windows.iloc[:, -1].values - means.values) / stds.values

# create torch tensors
x_windows = torch.tensor(x_windows, dtype=torch.float32)
y_labels = torch.tensor(y_labels, dtype=torch.float32)

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

# x_train, x_test = torch.utils.data.random_split(price_sequences, [train_size, test_size])
# y_train, y_test = torch.utils.data.random_split(labels, [train_size, test_size])

def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    if split == 'train':
        x, y = x_train, y_train
    else:
        x, y = x_test, y_test
    ix = torch.randint(len(y) - block_size, (batch_size,))
    x = torch.stack([x[i] for i in ix]) #torch.stack -> concatenates a sequence of tensors along a new dimension
    y = torch.stack([y[i] for i in ix])
    
    return x, y

@torch.no_grad() # when we're not doing back propagation is a good practice to use this decorator
def estimate_loss():
    out = {}
    model.eval() # we put it in evaluation mode (for dropout and batchnorm, usually)
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
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(block_size, block_size, bias=False)
        self.periodic = nn.Linear(block_size, block_size, bias=False)
        self.proj = nn.Linear(2, embeddings, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T = x.shape # B is the number of instances in the Barch; T is the number of prices (Tokens) we use as an input, defined by block_size
        time_linear=self.linear(x) # (B,T) 
        time_linear = time_linear.unsqueeze(-1) # (B,T,1)
        time_periodic=torch.sin(self.periodic(x)) # (B,T) | Apply the periodic layer and then the sin function
        time_periodic = time_periodic.unsqueeze(-1) # (B,T,1)
        out = torch.cat([time_linear, time_periodic], dim=-1) # (B,T,2) Concatenate the two tensors in the last dimension
        out = self.proj(out) # (B,T,2) @ (2,embeddings) -> (B,T,embeddings)
        return out

class Head(nn.Module):
    """One head of self-attention """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embeddings, head_size, bias=False)
        self.query = nn.Linear(embeddings, head_size, bias=False)
        self.value = nn.Linear(embeddings, head_size, bias=False)
        self.densek = nn.Linear(head_size, dense_layer, bias=False)
        self.denseq = nn.Linear(head_size, dense_layer, bias=False)
        self.densev = nn.Linear(head_size, dense_layer, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size * 2, block_size * 2)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # The input x is a tensor of shape (B,T,embeddings), which is the output of the time2vector module
        B, T, C = x.shape # B is the number of instances in the Barch;
                          # T is the number of prices (Tokens) we use as an input, defined by block_size;
                          # C = embeddings
        # Apply the key and query layers, and then a dense layer for each of them
        k=self.key(x) # (B,T,embeddings) @ (embeddings,head_size) -> (B,T,head_size)
        k=self.densek(k) # (B,head_size) @ (head_size,dense_layer) -> (B,T,dense_layer)
        q=self.query(x) # (B,T,embeddings) @ (embeddings,head_size) -> (B,T,head_size)
        q=self.denseq(q) # (B,head_size) @ (head_size,dense_layer) -> (B,T,dense_layer)
        
        # Compute the weights of the attention
        wei = q @ k.transpose(-2,-1) *C**-0.5 # (B,T,dense_layer) @ (B,dense_layer,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T) | Mask the upper triangular part of the matrix
        wei = F.softmax(wei, dim=-1) # (B,T,T) | Apply the softmax function to the last dimension to normalize the weights
                                     # so thy represent the weights of the attention
        # Apply dropout to the weights. This turns off (makes it 0) some of the weights randomly
        # It allows to avoid overfitting
        wei = self.dropout(wei) # (B,T,T)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,head_size) | Apply the value layer
        v = self.densev(v) # (B,T,dense_layer) | Apply the dense layer of the value
        out = wei @ v # (B,T,T) @ (B,T,dense_layer) -> (B,T,dense_layer) | Multiply the weights by the values
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, head_size):
        # We use super() to call the __init__() method of the parent class nn.Module
        # This is necessary because we are overriding the __init__() method of the parent class (nn.Module)
        super().__init__()
        # Initialize a list of heads, a projection layer, and a dropout layer
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads*dense_layer, embeddings) # 2 because we had 2 channels from the Time2Vector module
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Apply each head to the input x. Concatenate the outputs of the heads in the last dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B,T,num_heads*dense_layer)
        # This dropout is applied to the output of the heads
        # It allows to regularize the output of the heads
        out = self.dropout(self.proj(out)) # (B,T,num_heads*dense_layer) @ (num_heads*dense_layer,embeddings) -> (B,T,embeddings)
        return out 

class FeedForward(nn.Module):
    """ a simple layer followed by non-linearity """
    def __init__(self):
        super().__init__() # We use super() to call the __init__() method of the parent class nn.Module
        # Define a sequential aplication of layers.
        # The first layer is a linear layer, followed by a ReLU activation function, 
        # followed by a linear layer, followed by a dropout layer
        self.net = nn.Sequential(
            nn.Linear(embeddings, dense_layer), # (B,T,embeddings) @ (embeddings,dense_layer) -> (B,T,dense_layer)
            nn.ReLU(),
            nn.Linear(dense_layer, embeddings),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x) # (B,T,embeddings)
    

class Block(nn.Module):
    """ A single block of the transformer. We will use several of these blocks in the model """
    def __init__(self, block_size, n_head):
        # Initialize the parent class nn.Module
        super().__init__()
        # Define each head of the multi-head attention
        # It ensures that the size of the heads is a multiple of the number of heads
        head_size = block_size // n_head
        # Define the modules of the block based on the classes we defined above
        self.ttv = Time2Vector() 
        self.sa = MultiHeadAttention(n_head, head_size) 
        self.ffwd = FeedForward()
        # Define two layer normalization layers
        self.ln1 = nn.LayerNorm(block_size * embeddings)
        self.ln2 = nn.LayerNorm(block_size * embeddings)
        
    def forward(self, x):
        # Residual connection -> We simmply add the input to the output of the time2vector module
        x = torch.add(x.unsqueeze(-1), self.ttv(x), alpha=1) # (B,T,1) + (B,T,embeddings) -> (B,T,embeddings)
        # 1st normalization layer -> We apply layer normalization to the output of the time2vector module
        x = self.ln1(x.reshape(-1, block_size * embeddings)).reshape(-1, block_size, embeddings)
        # Multi-head attention
        x = torch.add(x, self.sa(x), alpha=1) # (B,T,embeddings)
        # 2nd normalization layer -> We apply layer normalization to the output of the multi-head attention
        x = self.ln2(x.reshape(-1, block_size * embeddings)).reshape(-1, block_size, embeddings)
        # Residual connection -> We add the input to the output of the multi-head attention
        # I guess this adds more strength to the signal provided by the key prices
        x = torch.add(x, self.ffwd(x), alpha=1) # (B,T,embeddings)
        return x

class Model(nn.Module):
    # Finally, this is the model that we will train, joining all the blocks together
    def __init__(self):
        # Initialize the parent class nn.Module
        super().__init__()
        # Define the blocks of the model
        # We use * to unpack the list of blocks
        # This allows us to pass a list of blocks sequentially to the nn.Sequential module
        # self.blocks = nn.Sequential(*[Block(block_size, n_head=n_head) for _ in range(n_layer)]) # (B,T,2)
        self.block = Block(block_size, n_head=n_head)
        # Convolutions to get the desired output dimension
        self.conv1 = nn.Linear(embeddings,1) 
        self.conv2 = nn.Linear(block_size,1) # (B,T,1) @ (T,1) -> (B,T,1)
 
    def forward(self, x, targets=None):
        B, T = x.shape # idx and targets is (B,T) 
        x = self.block(x) # (B,T,embeddings)
        # First convolution to get the desired output dimension
        x = self.conv1(x) # (B,T,embeddings) @ (embeddings,1) -> (B,T,1)
        # Erase the last dimension
        x = x.squeeze(-1) # (B,T)
        # Second convolution to get the desired output dimension
        x = self.conv2(x) # (B,T) @ (T,1) -> (B,1)
        # Erase the last dimension. Final output will be a vector of size B.
        # Each element of the vector is the prediction for each instance
        x = x.squeeze(-1) # (B)
        
        # If targets is not None, we compute the loss
        if targets is None:
            loss = None
        else:
            B = x.shape[0]
            targets = targets.view(B) #targets.view(B*T)
            loss = F.mse_loss(x, targets)

        return x, loss
    

    
model = Model()
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

# We'll transform the predictions back to the original scale
means = torch.tensor(means, dtype=torch.float32)
stds = torch.tensor(stds, dtype=torch.float32)

means_train, means_test = means[:train_size], means[train_size:]
stds_train, stds_test = stds[:train_size], stds[train_size:]

means_test = means_test.to(device)
means_train = means_train.to(device)
stds_test = stds_test.to(device)
stds_train = stds_train.to(device)

# Run the model on the test and train data. Discard the loss.
y_pred_test, _ = model(x_test)
y_pred_train, _ = model(x_train)

# Convert the predictions back to the original scale
prices_pred_test = y_pred_test * stds_test + means_test
prices_pred_train = y_pred_train * stds_train + means_train

prices_labels_test = y_test * stds_test + means_test
prices_labels_train = y_train * stds_train + means_train

#convert to numpy arrays
prices_pred_test = prices_pred_test.cpu().detach().numpy()
prices_pred_train = prices_pred_train.cpu().detach().numpy()

#Calculate the absolute error of each prediction
errors_test = prices_pred_test - prices_labels_test
errors_train = prices_pred_train - prices_labels_train

#plot the predictions
plt.figure(figsize=(12,6))
plt.plot(prices_pred_test, label='predictions')
plt.plot(prices_labels_test, label='labels')
plt.legend()
