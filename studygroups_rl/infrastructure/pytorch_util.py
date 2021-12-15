from typing import Union

import torch
from torch import nn
from typing import Tuple

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    """
        Builds a feedforward neural network
        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network
            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer
            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer
        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)


def build_ae(
    input_size:int,
    hidden_size:int,
    n_hidden:int,
    latent_size:int,
    activation: Activation = 'relu',
    output_activation: Activation = 'identity'

):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    encoder_layers = []
    in_size = input_size

    for _ in range(n_hidden):
        encoder_layers.append(nn.Linear(in_size, hidden_size))
        encoder_layers.append(activation)
        in_size = hidden_size
    encoder_layers.append(nn.Linear(in_size, latent_size))
    encoder_layers.append(output_activation)

    
    decoder_layers = []
    in_size = latent_size
    for _ in range(n_hidden):
        decoder_layers.append(nn.Linear(in_size, hidden_size))
        decoder_layers.append(activation)
        in_size = hidden_size
    decoder_layers.append(nn.Linear(in_size, input_size))
    decoder_layers.append(output_activation)


    return encoder_layers, decoder_layers

def build_conv_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        channel_in: int,
        channel_out: int,
        kernel_size: Tuple[int, int],
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []
    in_size = input_size

    layers.append(nn.Conv1d(channel_in, channel_out, kernel_size))
    layers.append(activation)

    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)

    return nn.Sequential(*layers)

device = torch.device("cpu")

class AE(nn.Module):
    def __init__(self, 
            input_size:int,
            hidden_size:int,
            n_hidden:int,
            latent_size:int,
            activation: Activation = 'relu',
            output_activation: Activation = 'identity',
            dropout: bool =True,
            dropout_rate: float = 0.1
    ):
        super().__init__()
        self.encoder_layers, self.decoder_layers = build_ae(input_size,
                                        hidden_size,
                                        n_hidden,
                                        latent_size,
                                        activation,
                                        output_activation)
        self.encoder = nn.Sequential(*self.encoder_layers)
        self.decoder = nn.Sequential(*self.decoder_layers)
        self.encoder_dropout = nn.Dropout(dropout_rate)
        self.decoder_dropout = nn.Dropout(dropout_rate)
        self.dropout = dropout

                                        
        
    def forward(self, x):
        
        if self.dropout:
            x = self.encoder_dropout(x)
        encoded = self.encoder(x)
        if self.dropout:
            encoded = self.decoder_dropout(encoded)
        decoded = self.decoder(encoded)
        return decoded
    
    def forward_eval(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



def init_gpu(use_gpu=True, gpu_id=0):
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()