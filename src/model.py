import math
import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import torchvision.transforms as transforms
import torch.nn.functional as F
from torchtext.datasets import PennTreebank
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import matplotlib.pyplot as plt


from torch.nn import TransformerEncoder, TransformerEncoderLayer
import inspect
import time
from tempfile import TemporaryDirectory
import util
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int = 200, d_model: int = 32, nhead: int = 32, d_hid: int = 32,
                 nlayers: int = 2, dropout: float = 0.5, model_id = None):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)
        self.id = model_id
        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask = None) :
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


train_iter = PennTreebank(split='train')
print(train_iter)
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>'])

def data_process(raw_text_iter):
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# ``train_iter`` was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = PennTreebank()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

def batchify(data, bsz: int):
    """Divides the data into ``bsz`` separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Arguments:
        data: Tensor, shape ``[N]``
        bsz: int, batch size

    Returns:
        Tensor of shape ``[N // bsz, bsz]``
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
val_data = batchify(val_data, eval_batch_size)
print(train_data.shape)
test_data = batchify(test_data, eval_batch_size)

bptt = 35
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability
epoch = 0 # this is a dummy count
model_A = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, model_id = "A").to(device)
model_B = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout, model_id = "B").to(device)


criterion = nn.CrossEntropyLoss()
lr = 2.5  # learning rate
best_val_loss = float('inf')
epochs = 1

def get_batch(source, i: int):
    """
    Args:
        source: Tensor, shape ``[full_seq_len, batch_size]``
        i: int

    Returns:
        tuple (data, target), where data has shape ``[seq_len, batch_size]`` and
        target has shape ``[seq_len * batch_size]``
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

def shuffle_data(data, seed=None):
    """
    Shuffle the columns of the input tensor to introduce randomness
    in the order of sequences while preserving the internal order within each sequence,
    with an optional seed for controlled randomness.

    Args:
        data: Tensor, shape [seq_len, batch_size]
        seed: Optional[int], a seed for the random number generator for reproducibility

    Returns:
        Shuffled data with the same shape.
    """
    seq_len, batch_size = data.shape
    if seed is not None:
        torch.manual_seed(seed)  # Set the random seed if provided
    
    # Generate a permutation of indices and use it to shuffle the columns
    indices = torch.randperm(batch_size)
    shuffled_data = data[:, indices]
    # print(indices)
    return shuffled_data

def train(model: nn.Module, lr = lr, epoch = 0, optimizer = None, scheduler = None, alpha = -1, train_data =train_data) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    total_correct = 0
    total_samples = 0
    start_time = time.time()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        output = model(data)
        output_flat = output.view(-1, ntokens)
        loss = criterion(output_flat, targets)

        _, predicted = torch.max(output_flat, 1)
        total_correct += (predicted == targets).sum().item()
        total_samples += targets.size(0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_accuracy = total_correct / total_samples
            ppl = math.exp(cur_loss)
            # record the plot when at the last period of this epoch
            if log_interval > num_batches - batch:
                writer.add_scalar(f"Loss of {model.id}/train", cur_loss, epoch)
                writer.add_scalar(f"Accuracy of {model.id}/train", cur_accuracy, epoch)
            message = f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '\
                      f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '\
                      f'loss {cur_loss:5.2f} | ppl {ppl:8.2f} | accuracy {cur_accuracy*100:.2f}%'
            if alpha > -1:
                message += f' | alpha {alpha}'
            print(message)
            total_loss = 0
            total_correct = 0  # Reset for the next interval
            total_samples = 0
            start_time = time.time()
        # iteration += 1

def evaluate(model: nn.Module, eval_data):
    model.eval()
    total_loss = 0.
    total_accuracy = 0.
    total_samples = 0
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            seq_len = data.size(0)
            output = model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += seq_len * criterion(output_flat, targets).item()
            _, predicted = torch.max(output_flat, dim=1)
            total_accuracy += (predicted == targets).sum().item()
            total_samples += targets.size(0)
    return total_loss / (len(eval_data) - 1), total_accuracy / total_samples

def roll_iter(model, alpha = -1, early_stop = epochs, epoch2record = None, optimizer_in = None, seed = 42, shuffle = False):
    best_val_loss = float('inf')
    optimizer = None
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, "best_model_params.pt")
        if optimizer_in is None:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            optimizer = optimizer_in
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
        data = train_data
        if shuffle:
            data = shuffle_data(train_data, seed)
        for epoch in range(1, epochs + 1):
            epoch_start_time  = time.time()
            train(model, epoch = epoch, optimizer=optimizer, scheduler=scheduler, alpha = alpha, train_data=data)
            val_loss, val_acccuracy = evaluate(model, val_data)
            val_ppl = math.exp(val_loss)
            writer.add_scalar(f"Loss of {model.id}/valid", val_loss, epoch)
            writer.add_scalar(f"val_ppl  of {model.id}/valid", val_ppl, epoch)
            writer.add_scalar(f"Accuracy  of {model.id}/valid",val_acccuracy, epoch )
            elapsed = time.time() - epoch_start_time
            print('-' * 89)
            message = f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ' \
                    f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}'

            # Conditionally add information about alpha
            if alpha > -1:
                message += f' | alpha {alpha}'
            print(message)
            print('-' * 89)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_params_path)

            scheduler.step()
            if epoch2record is not None: 
                save_checkpoint(model=model, epoch = epoch)
            else:
                if alpha == -1 and model.id is not None:
                    # torch.save(model.state_dict(), f"checkpoint/model_{model.id}/checkpoint_{model.id}_{epoch}.pt")
                    save_checkpoint(model=model, epoch = epoch)
            
            if epoch == early_stop:
                break
        model.load_state_dict(torch.load(best_model_params_path)) # load best model states

def save_checkpoint(model, epoch):
    model_dir = f"checkpoint/model_{model.id}"

    # Check if the parent directory exists, if not, create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Save the model state dict
    model_path = os.path.join(model_dir, f"checkpoint_{model.id}_{epoch}.pt")
    torch.save(model.state_dict(), model_path)



# linear mode connectivity transformer
def get_network_parameters(model):
    params = []
    for param in model.parameters():
        params.append(param.clone())  # .detach()?
    return params

def set_network_parameters(model, params):
    for model_param, input_param in zip(model.parameters(), params):
        model_param.data.copy_(input_param)

def interpolated_network(model_A, model_B, training = False):
    epsilon = 0.1
    alpha = 0
    net1_params = get_network_parameters(model_A)
    net2_params = get_network_parameters(model_B)
    alpha_values = []
    losses = []
    assert type(model_A) == type(model_B), "Networks must have the same architecture"
    
    config = {"ntoken" : len(vocab),  # size of vocabulary
                "d_model" : 200,  # embedding dimension
                "d_hid" : 200,# dimension of the feedforward network model in ``nn.TransformerEncoder``
                "nlayers" : 2,  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
                "nhead" : 2,
                "dropout": 0.2, }  # number of heads in ``nn.MultiheadAttention``}
    # Create a new network with the same architecture
    interpolated_net = type(model_A)(**config).to(device)
    interpolated_params = get_network_parameters(interpolated_net)

    def _interpolate_network(alpha):

        # Interpolate the parameters
        for p1, p2, p_interpolated in zip(net1_params, net2_params, interpolated_params):
            p_interpolated.data.copy_(alpha * p1 + (1 - alpha) * p2)

        set_network_parameters(interpolated_net, interpolated_params)
        return interpolated_net
    sup_alpha = 0
    sup_loss = float('-inf')        
    cur_loss = float('-inf')
    cur_ppl =  float('-inf')
    mode = "train"
    while alpha < 1.01:
        interpolated_network = _interpolate_network(alpha)
        interpolated_network.id = str(alpha)

        if training :
            # roll_iter(interpolated_network, alpha = alpha)
            cur_loss, cur_ppl = run_result(interpolated_network, alpha, train=True)
            # pass
        else:
            # test_loss, test_ppl = run_result(interpolated_network, alpha = alpha)
            # writer.add_scalar(f"loss/alpha", test_loss, alpha)
            # writer.add_scalar(f"ppl/alpha", test_ppl, alpha)
            mode = "test"
            cur_loss, cur_ppl = run_result(interpolated_network, alpha, train=False)
        # writer.add_scalar()
        # writer.add_scalar(f"loss_{mode}/alpha", alpha, cur_loss)
        # writer.add_scalar(f"ppl_{mode}/alpha", cur_ppl, alpha)
        alpha_values.append(alpha)
        losses.append(cur_loss)
        if cur_loss > sup_loss:
            sup_loss = cur_loss
            sup_alpha = alpha
        alpha = alpha+epsilon
        plt.figure(figsize=(10, 5))
        plt.plot(alpha_values, losses, marker='o', linestyle='-', color='b')
        plt.title('Loss vs Alpha')
        plt.xlabel('Alpha')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
    return sup_alpha, sup_loss

def load_checkpoint(model, path = None, start_point = None):
    state_dict = None
    assert os.path.exists(path=path)
    if start_point is not None: 
        path = f"checkpoint/model_{model.id}/checkpoint_{model.id}_{start_point}.pt"
        if os.path.exists(path=path):
            state_dict = torch.load(path, map_location=torch.device('cpu'))
        else:
            return None
    else: 
        state_dict = torch.load(path, map_location=torch.device('cpu'))
    # Update the model's state dictionary
    model.load_state_dict(state_dict)
    return model

def run_result(model, alpha = -1, train = False):
    # roll_iter(model)
    loss = None
    ppl = None
    mode = "Test"
    data = test_data
    if train:
        data = train_data
        mode = "Train"
    loss, accuracy = evaluate(model, data)
    ppl = math.exp(loss)
    writer.add_scalar(f"Loss/{mode}", loss)
    writer.add_scalar(f"ppl/{mode}", ppl)
    print('=' * 89)
    message = f'model {model.id}| {mode} loss {loss:5.2f} | 'f'{mode} ppl {ppl:8.2f}'
    if alpha > -1:
        message += f' | alpha {alpha}'
        writer.add_scalar(f"Loss_alpha/{mode}", loss, alpha)
        writer.add_scalar(f"ppl_alpha/{mode}", ppl, alpha)
    print(message)
    # print()
    print('=' * 89)
    return loss, ppl

def continue_training(start_point, model):
    loaded_trained = load_checkpoint(model, start_point = start_point)
    roll_iter(loaded_trained)

def analysis4training(model_A, model_B, start_point = 4):
    # assume model A and model are all trained, with checkpoints prepared
    trained_A = load_checkpoint(model_A, start_point = start_point)
    trained_B = load_checkpoint(model_B, start_point = start_point)
    interpolated_network(trained_A, trained_B, training = True)
    
def analysis4test(model_A, model_B):
    trained_A = load_checkpoint(model_A, start_point = 8)
    trained_B = load_checkpoint(model_B, start_point = 8)
    interpolated_network(trained_A, trained_B, training = False)

# instability analysis at initialization(nor pretrained steps)
def analysis_init(model):
#    use the same network weight, but different optimizer
    _net_copy1 = get_network_parameters(model)
    _net_copy2 = get_network_parameters(model)
    config = {"ntoken" : len(vocab),  # size of vocabulary
                "d_model" : 200,  # embedding dimension
                "d_hid" : 200,# dimension of the feedforward network model in ``nn.TransformerEncoder``
                "nlayers" : 2,  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
                "nhead" : 2,
                "dropout": 0.2, }  # number of heads in ``nn.MultiheadAttention``}
    # Create a new network with the same architecture
    net_copy1 = type(model)(**config).to(device)
    net_copy2 = type(model)(**config).to(device)
    
    set_network_parameters(net_copy1, _net_copy1)
    set_network_parameters(net_copy2, _net_copy2)
    net_copy1.id = "init_copy1"
    net_copy2.id = "init_copy2"

    net1 = load_checkpoint(net_copy1, start_point=epochs)
    net2 = load_checkpoint(net_copy2, start_point=epochs)
    
    # check is already saved, then load from checkpoint
    if net1 is not None:
        net_copy1 = net1
    else: 
        roll_iter(net_copy1, seed=41, shuffle=True)

    if net2 is not None:
        net_copy2 = net2
    else:
        roll_iter(net_copy2, seed=40, shuffle=True)

    interpolated_network(net_copy1, net_copy2, training=False)
    interpolated_network(net_copy1, net_copy2, training=True)
    # run_result()
    # pass
# roll_iter(model_A)
analysis_init(model_A)
# run_result(model_A)
# torch.save(model_A.state_dict(), "checkpoint/model_A/checkpoint_A_complete.pt")
# roll_iter(model_B)
# run_result(load_checkpoint(model_B, start_point = 8))
# torch.save(model_B.state_dict(), "checkpoint/model_B/checkpoint_B_complete.pt")
# epsilon_error = 0.1
# analysis4training(model_A, model_B)
# analysis4test(model_A, model_B)

# analysis_init(model=model_A)

# model_loadA = load_checkpoint(model_A, "checkpoint/model_A/checkpoint_A.pt")
# model__loadB = load_checkpoint(model_B, "checkpoint/model_A/checkpoint_A.pt")
# sup_alpha, sup_loss = interpolated_network(model_A, model__B)
# print(f'sup alpha:  {sup_alpha} sup loss: {sup_loss}')