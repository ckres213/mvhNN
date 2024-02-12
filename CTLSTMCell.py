import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class CTLSTMCell(nn.Module):
    def __init__(self, lstm_units, scale=0.1, device=None):
        """
        Initializes a CTLSTMCell object.

        Parameters:
        - lstm_units (int): Dimensionality of the hidden state.
        - scale (float): Hyperparameter controlling decay computation.
        - device (str): Device to place the model on
        """
        super(CTLSTMCell, self).__init__()

        device = device or 'cpu'
        self.device = torch.device(device)

        self.lstm_units = lstm_units
        
        '''
        (lstm_units * 2): input to the linear layer is the concatenation of the current input (event_type_emb_i) and the previous hidden state (hidden).

        (lstm_units * 7): The seven different gates are the input_gate, forget_gate, output_gate, cell_input_gate (z), input_bar_gate, forget_bar_gate and decay_gate

        The bias=True argument indicates that the layer includes bias terms.
        '''
        
        self.linear = nn.Linear(lstm_units * 2, lstm_units * 7, bias=True)
        
        #cb
        self.scale = scale

    def forward(self, event_type_emb_i, hidden_t__i_minus_1, cell_t__i_minus_1, cell_bar_i_minus_1):

        dim_input = event_type_emb_i.dim() - 1
        
        # input_gate for the ith event
        input_i = torch.cat((event_type_emb_i, hidden_t__i_minus_1), dim=dim_input)
        
        # output_gate of the linear transformation
        output_i = self.linear(input_i)

        # Splitting output_gate into the 7 gates (cell_input_gate is z in Mei)
        input_gate_i, forget_gate_i, output_gate_i, cell_input_gate_i, \
        input_bar_gate_i, forget_bar_gate_i, \
        decay_gate_i = output_i.chunk(7, dim_input)
             
        # Applying transformations to the dates
        input_gate_i = torch.sigmoid(input_gate_i)
        forget_gate_i = torch.sigmoid(forget_gate_i)
        output_gate_i = torch.sigmoid(output_gate_i)
        cell_input_gate_i = torch.tanh(cell_input_gate_i)
        input_bar_gate_i = torch.sigmoid(input_bar_gate_i)
        forget_bar_gate_i = torch.sigmoid(forget_bar_gate_i)
        decay_gate_i = F.softplus(decay_gate_i, beta=self.scale)

        cell_i = forget_gate_i * cell_t__i_minus_1 + input_gate_i * cell_input_gate_i
        cell_bar_i = forget_bar_gate_i * cell_bar_i_minus_1 + input_bar_gate_i * cell_input_gate_i

        return cell_i, cell_bar_i, decay_gate_i, output_gate_i

    def decay(self, cell_i, cell_bar_i, decay_gate_i, output_gate_i, interarrivals):
        # interarrivals is the ith event in each sequence in batch
        if interarrivals.dim() < cell_i.dim():
            # Makes interarrivals column vector and repeats untill same size as cell
            interarrivals = interarrivals.unsqueeze(cell_i.dim() - 1).expand_as(cell_i)
        
        # Element-wise multiplication of decay_gate_i and interarrivals
        cell_t__i = cell_bar_i + (cell_i - cell_bar_i) * torch.exp(
            -decay_gate_i * interarrivals)
        
        hidden_t__i = output_gate_i * torch.tanh(cell_t__i)

        return cell_t__i, hidden_t__i
