import torch
import CTLSTMCell
import torch.nn as nn
import pdb


class Conttime(nn.Module):
    def __init__(self, dim_process, lstm_units, scale = 0.1, learn_rate = 0.01):
        self.dim_process = dim_process
        self.lstm_units = lstm_units
        # Scaling parameter in the softplus function
        self.scale = scale

        super(Conttime, self).__init__()
        # Embedding layer for event type indices
        # Creates map from self.n_types + 1 categories and stores each category as a vector of dimension self.hid_dim such that the categories have a continuous representation
        self.emb = nn.Embedding(self.dim_process + 1, self.lstm_units)
        
        # Creates R^{hidden_dim * 2} -> R^{hidden_dim * 7} linear map
        self.lstm_cell = CTLSTMCell.CTLSTMCell(lstm_units, scale)
        
        # Creates R^{hid_dim} -> R^{n_types} linear map. Is equivalent to the w_k^T vector for all k
        #self.hidden_lambda = nn.Linear(self.lstm_units, self.dim_process)
        
        self.hidden_lambda = nn.Linear(self.lstm_units, self.dim_process, bias=False)

        # Adam optimizer for training with the given learning rate
        # Parameters are emb.weight, lstm_cell.linear.weight, lstm_cell.linear.bias, hidden_lambda.weight, hidden_lambda.bias
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

    def train_batch(self, interarrival_seqs, event_types_seqs, sim_diffs_seqs, total_duration_seqs, length_seqs, sim_diffs_index):
        # Forward pass to get hidden states and cell states
        hidden_states, cell_gates, cell_bar_gates, decay_gates, output_gates = self.forward(event_types_seqs, interarrival_seqs)
        
        # Compute likelihood components and total likelihood
        part_one_likelihood, part_two_likelihood, sum_likelihood = self.conttime_loss(hidden_states, 
                                                                                      cell_gates, 
                                                                                      cell_bar_gates,
                                                                                      decay_gates, 
                                                                                      output_gates,
                                                                                      event_types_seqs, 
                                                                                      sim_diffs_seqs, 
                                                                                      total_duration_seqs,
                                                                                      length_seqs, 
                                                                                      sim_diffs_index)
        # Calculate negative log-likelihood loss for the batch
        loss = -(torch.sum(part_one_likelihood - part_two_likelihood))
        
        # Backpropagate gradients through the computational graph
        # This line computes the gradients of the loss with respect to all tensors that have requires_grad=True
        loss.backward()

        # Update model parameters using the computed gradients
        # Here, we perform a parameter update step using the optimizer (e.g., SGD, Adam).
        # The optimizer adjusts the parameters of the model based on the computed gradients
        # in order to minimize the loss function.
        self.optimizer.step()

        # Clear the gradients of model parameters
        self.optimizer.zero_grad()
        
        return loss

    def forward(self, event_types_seqs, interarrival_seqs):
        # Saves number of sequences and sequence length
        num_seqs, length_seqs = interarrival_seqs.shape
                
        # Create num_seqs x self.lstm_units tensors to store the initialized hidden state (h(0)), cell_t (c(0)), and cell_bar (\overline{c}_0). 
        self.hidden_t__i = torch.zeros(num_seqs, self.lstm_units, dtype=torch.float32)
        self.cell_t__i = torch.zeros(num_seqs, self.lstm_units, dtype=torch.float32)
        self.cell_bar_i = torch.zeros(num_seqs, self.lstm_units, dtype=torch.float32)
        
        # Lists to store LSTM states
        hidden_list, cell_list, cell_bar_list, decay_list, output_list = [],[],[],[],[]
       
        # Iterate through the events in the sequence
        for i in range(length_seqs - 1):
            # Extract the embedded "type" for the ith event in each sequence in batch (k in equations (5a) through (5d)
            event_type_emb_i = self.emb(event_types_seqs[:,i])
            
            # Using prev calculated cells/states for update formulas
            self.hidden_t__i_minus_1 = torch.clone(self.hidden_t__i)
            self.cell_t__i_minus_1 = torch.clone(self.cell_t__i)
            self.cell_bar_i_minus_1 = torch.clone(self.cell_bar_i)
            
            # Forward pass through LSTM to get gates
            # cell_bar_i has self. due to its initalization outside the loop. 
            cell_i, self.cell_bar_i, decay_i, output_i = self.lstm_cell.forward(event_type_emb_i, self.hidden_t__i_minus_1, self.cell_t__i_minus_1, self.cell_bar_i_minus_1)
            
            # Uses gates to find c(t_i) and h(t_i)
            self.cell_t__i, self.hidden_t__i = self.lstm_cell.decay(cell_i, self.cell_bar_i, decay_i, output_i, interarrival_seqs[:,i+1])
            
            # Appends output to lists
            hidden_list.append(self.hidden_t__i)
            cell_list.append(cell_i)
            cell_bar_list.append(self.cell_bar_i)
            decay_list.append(decay_i)
            output_list.append(output_i)
        
        # Takes individual lists and stacks them into a higher dimension.
        # This is a max_len x batch_size x lstm_units tensor. Or the hidden state at each event time for every sequence in the batch
        hidden_states = torch.stack(hidden_list)
        
        # Transforms gates from a list of length max_len with each object made up of batch_size lists of lstm_units to a tensor with dimensions max_len x batch_size x lstm_units
        cell_gates = torch.stack(cell_list)
        cell_bar_gates = torch.stack(cell_bar_list)
        decay_gates = torch.stack(decay_list)
        output_gates = torch.stack(output_list)
        
        return hidden_states, cell_gates, cell_bar_gates, decay_gates, output_gates

    def conttime_loss(self, hidden_states, cell_gates, cell_bar_gates, decay_gates, output_gates, event_types_seqs, sim_diffs_seqs, total_duration_seqs, length_seqs,
                      sim_diffs_index):
        # Calculates likelihood values for each sequence in batch
        
        # Number of batches
        batch_size = event_types_seqs.shape[0]
        
        # Length of simulated sequence
        sim_len = sim_diffs_index.shape[1]
        
        # Initalizing Likelihood
        # The sum of the log-intensities of the events that happened, at the times they happened
        part_one_likelihood = torch.zeros(batch_size)
        
        sum_likelihood = torch.zeros(batch_size)
        
        # Calculates intensity for each sequence in batch at each event
        # self.hidden_lambda(hidden_states) is a max_len x batch_size x dim_process tensor
        type_intensity = torch.nn.functional.softplus(self.hidden_lambda(hidden_states)).transpose(0,1)
        
        # Looping over sequences in batch
        for idx in range(batch_size):
            # Event types for a sequence
            event_seq = event_types_seqs[idx]
            # Number of events in sequence 
            length_seq = length_seqs[idx]
                       
            # Takes unpadded idx^th sequence (actual realizations), and extracts type intensity of the event type that occured. Log transforms the intensities and sums
            
            part_one_likelihood[idx] = torch.sum(
                torch.log(
                    type_intensity[idx, 
                                   torch.arange(length_seq), 
                                   event_seq[1:length_seq+1]]
                )
            )
            
            # Takes unpadded idx^th sequence (actual realizations), and extracts the type intnsities at each event. Sums intensities at each event, log transforms then sums     
            
            sum_likelihood[idx] = torch.sum(
                torch.log(
                    torch.sum(
                        type_intensity[idx, 
                                       torch.arange(length_seq), 
                                       :], 
                        dim=-1
                    )
                )
            )
        
        # Initalize simulated gate values
        sim_cell_gates = []
        sim_cell_bar_gates = []
        sim_decay_gates = []
        sim_output_gates = []
               
        # Looping over sequences in batch
        for idx in range(batch_size):
            
            # Finds value of gates of the original sequences at simulated index points???
            layer_cell_gates = cell_gates[sim_diffs_index[idx], idx, :]
            sim_cell_gates.append(layer_cell_gates)
           
            layer_cell_bar_gates = cell_bar_gates[sim_diffs_index[idx], idx, :]
            sim_cell_bar_gates.append(layer_cell_bar_gates)
            
            layer_decay_gates = decay_gates[sim_diffs_index[idx], idx, :]
            sim_decay_gates.append(layer_decay_gates)
            
            layer_output_gates = output_gates[sim_diffs_index[idx], idx, :]
            sim_output_gates.append(layer_output_gates)
        
        # Transforms sim_gates from a list of length batch_size with each object made up of sim_len lists of lstm_units to a tensor with dimensions sim_len x batch_size x lstm_units
        sim_cell_gates = torch.stack(sim_cell_gates).transpose(0,1)
        sim_cell_bar_gates = torch.stack(sim_cell_bar_gates).transpose(0,1)
        sim_decay_gates = torch.stack(sim_decay_gates).transpose(0,1)
        sim_output_gates = torch.stack(sim_output_gates).transpose(0,1)
        
        sim_hidden_list = []
        
        # Looping over each simulated event
        for idx in range(sim_diffs_seqs.shape[1]):
            # Calculates simulated hidden state at the idx^th event for each sequence in batch
            _, sim_hidden_t__i = self.lstm_cell.decay(sim_cell_gates[idx], 
                                                        sim_cell_bar_gates[idx],
                                                        sim_decay_gates[idx], 
                                                        sim_output_gates[idx], 
                                                        sim_diffs_seqs[:, idx])

            sim_hidden_list.append(sim_hidden_t__i)            
        
        # Transforms hidden_list from a list of length sim_len with each object made up of batch_size lists of lstm_units to a tensor with dimensions sim_len x batch_size x lstm_units
        sim_hidden_list = torch.stack(sim_hidden_list)        
        
        # Calculates simulated intensity at each simulated event time
        sim_intensity = torch.nn.functional.softplus(
            self.hidden_lambda(sim_hidden_list)
        ).transpose(0,1)
        
        # Integral of the total intensities over the observation interval [0, T]
        part_two_likelihood = torch.zeros(batch_size)
        
        # Looping over sequences in batch 
        for idx in range(batch_size):
            # Total sequence time / number of simulated events 
            coefficient = total_duration_seqs[idx] / sim_len
            
            # Sums the intensity values at each simulated event, and multiplies by the average time between events #cb
            part_two_likelihood[idx] = torch.sum(
                sim_intensity[idx, 
                              torch.arange(sim_len), 
                              :]
            ) * coefficient
        
        return part_one_likelihood, part_two_likelihood, sum_likelihood

if __name__ == "__main__":
    """
    """
