import torch
import torch.nn as nn
from torch.nn import functional as F

## convert to self-attention model 1:02:47
# hyperparameters
batch_size = 64 #32 # how many independent sequences will we process in parallel?
block_size = 256 #8 # what is the maximum context length for predictions?
max_iters = 5000 #3000: increse since learning rate is lower
eval_interval = 500 #300
learning_rate = 3e-4 #1e-3 #1e-2 self attention cant't tolerate very high learning rate
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 384#32#128
n_head = 6
n_layer = 6
dropout = 0.2 # 20% of intermediate calculations will be disabled/zeroed
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
## global variable vocab_size
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

## NOTE:I 1)load batch, randomly get id of batch size, and use id to get samples from original data
## 2) X and Y are of same size
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

## NOTE: 1) self-attention,
class Head(nn.Module):
    """ one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        ## tril is not a parameter of the module, is a buffer not a parameter
        ## lower triangular matrix
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k =self.key(x) # (B,T,C) -> (B,T,H)
        q = self.query(x) # (B,T,C) -> (B,T,H)
        ## compute atteintion scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B,T,H) @ (B,H,T) -> (B,T,T)
        ## decoder blocl, feature doesn't communicate with the past
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T), higher triangular mask
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C) -> (B,T,H)
        out = wei @ v # (B,T,T) @ (B,T,H) -> (B,T,H)
        return out

 ## NOTE: 2) multi-head self-attention, 
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        ## residual connection
        self.proj = nn.Linear(n_embed, n_embed) # n_embed = num_heads*head_size
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
## NOTE: 3) Feedforward for block 
class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """    
    ## feed-forward layer multiply num_embed by 4, according to the paper

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(), # non-linearity
            ## residual connection
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

## NOTE: 3) block 
## solve optimiation problem of deep NN, using residual connection, 1:29:00    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embed, n_head):
        # n_embed: embedding dimension, n_head, the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        ## pre-norm formulation, add normalization before the transformer, while the original paper add after
        self.ln1 = nn.LayerNorm(n_embed) # default dim=1. normalize features for each sample
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # x = self.sa(x)
        # x = self.ffwd(x)
        ## residual connection
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))        
        return x
        

## NOTE: code GPT model from previous simple bigram model
class GPT(nn.Module):

    def __init__(self):
        super().__init__()
        # version 1: each token directly reads off the logits for the next token from a lookup table
        # self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

        # version 2: identy embedding of each token followed by a linear layer
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        ## position of token in the sequence
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        # ## from token embedding to the logits
        # ## v1: self-attention
        # # self.sa_head = Head(n_embed)
        # ## v2: multi-head self-attention, kind of like a group convolution
        # self.sa_head = MultiHeadAttention(num_heads=4, head_size=n_embed//4) # i.e., 4 heads of 8-dim self-attention
        # self.ffwd = FeedForward(n_embed)
        ## v3: block
        # self.blocks = nn.Sequential(
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        #     Block(n_embed, n_head=4),
        # )
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) ## final layer normalization
        self.lm_head = nn.Linear(n_embed, vocab_size)


    def forward(self, idx, targets=None):
        B, T = idx.shape

        # version1: idx and targets are both (B,T) tensor of integers
        # logits = self.token_embedding_table(idx) # (B,T,C)
        ## version 2: add position embedding
        token_embed = self.token_embedding_table(idx) # (B,T,C): C: n_embed
        pos_embed = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = token_embed + pos_embed # (B, T, C)
        # x = self.sa_head(x) # apply one head of self-attention (B,T,C)
        # x = self.ffwd(x) # apply feed-forward layer (B,T,C)
        x = self.blocks(x) # apply feed-forward layer (B,T,C)
        x = self.ln_f(x) # B, T, C
        logits = self.lm_head(x) # (B,T,V): V: vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            ## posistion embedding only have block_size
            ## get the recent block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond) # self(idex), (B, T, V)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, V)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, V)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPT()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
