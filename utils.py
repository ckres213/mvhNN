import torch
import pickle
from torch.utils.data import DataLoader
import pdb


def open_pkl_file(path, description):
    # Opens previously saved data and accesses the specified dataset
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        data = data[description]
    
    # Initalizing lists
    interarrival_seqs = []
    event_types_seqs = []
    length_seqs = []
    
    # For a given sequence, the loop creates the tensors of interarrival times, event types and number of events and appends to seperate lists, and repeats this for all sequences
    for i in range(len(data)):
        length_seqs.append(len(data[i]))
        event_types_seqs.append(torch.LongTensor([int(event['event_type']) for event in data[i]]))
        interarrival_seqs.append(torch.FloatTensor([float(event['time_since_last_event']) for event in data[i]]))
    return interarrival_seqs, event_types_seqs, length_seqs


'''
def open_txt_file(path):
    f = open(path, 'r')
    data_file = f.readlines()
    f.close()
    time_duration = []
    length_seqs = []
    # total_time_list = []
    for line in data_file:
        data = line.split(" ")
        a_list = []
        previous = 0
        lens = 0
        for i in range(len(data)):
            if data[i] != "\n":
                a_list.append(float(data[i]) - previous)
                previous = float(data[i])
                lens += 1
        time_duration.append(torch.tensor(a_list))
        # total_time_list.append(previous)
        length_seqs.append(lens)
    return time_duration, length_seqs
'''

def get_index_txt(duration):
    type_list = []
    for i in range(len(duration)):
        a_list = torch.zeros(size=duration[i].shape, dtype=torch.long)
        type_list.append(a_list)
    type_list = torch.stack(type_list)
    return type_list


def padding_full(interarrival_seqs, event_types_seqs, length_seqs, dim_process):
    # Maximum number of events across any of the sequences
    max_len = max(length_seqs)
    
    # Number of sequences
    sequence_count = len(interarrival_seqs)
    
    # Initalizes tensor of padded sequences for interarrival and event types
    interarrival_seqs_padded = torch.zeros(size=(sequence_count, max_len+1))
    event_types_seqs_padded = torch.zeros(size=(sequence_count, max_len+1), dtype=torch.long)
    
    # Looping over each sequence
    for idx in range(sequence_count):
        # Add ith sequence of interarrival times to the tensor, leaving the first value as 0
        interarrival_seqs_padded[idx, 1:length_seqs[idx]+1] = interarrival_seqs[idx]
        
        # Make first value in ith event type sequence the dimension of the process (BOS event)
        event_types_seqs_padded[idx, 0] = dim_process
        # Add ith sequence of event types to the tensor
        event_types_seqs_padded[idx, 1:length_seqs[idx]+1] = event_types_seqs[idx]

    return interarrival_seqs_padded, event_types_seqs_padded


'''
def padding_length_seqs(interarrival_seqs, event_types_seqs, dim_process, length_seqs):
    interarrival_seqs = []
    type_lists = []
    length_seqs = []
    batch_size = len(interarrival_seqs)
    for i in range(batch_size):
        end = length_seqs
        while end <= interarrival_seqs[i].shape.__getitem__(-1):
            start = end - length_seqs
            interarrival_seqs_list = [0]
            type_list = [dim_process]
            interarrival_seqs_list = interarrival_seqs_list + interarrival_seqs[i][start:end].tolist()
            type_list = type_list + event_types_seqs[i][start:end].tolist()
            interarrival_seqs.append(interarrival_seqs_list)
            type_lists.append(type_list)
            length_seqs.append(length_seqs)
            end += 1
    interarrival_seqs = torch.tensor(interarrival_seqs)
    type_lists = torch.tensor(type_lists)
    return interarrival_seqs, type_lists, length_seqs
'''

def generate_simulation(interarrival_seqs, length_seqs):
    # Length of simulated sequence
    sim_len = max(length_seqs) * 5
    
    # Initalizing tensors
    # Difference between random time and most recent actual event
    sim_diffs_seqs = torch.zeros(interarrival_seqs.shape[0], sim_len)
    # Index of most recent actual event
    sim_diffs_index = torch.zeros(interarrival_seqs.shape[0], sim_len, dtype=torch.long)
    
    total_duration_seqs = []
    
    # Looping over each sequence in the batch
    for idx in range(interarrival_seqs.shape[0]):
        # Calculates actual event times for the sequence
        event_times_seq = torch.stack([torch.sum(interarrival_seqs[idx][:i]) for i in range(1, length_seqs[idx]+2)])
        
        # Duration that the sequence occurs on 
        duration_seq = event_times_seq[-1].item()
     
        total_duration_seqs.append(duration_seq)
        
        # Generate a sorted random sequence of time points within the total event duration
        sim_times_seq, _ = torch.sort(torch.empty(sim_len).uniform_(0, duration_seq))
        
        # Initalized difference between random time and most recent actual event 
        sim_diffs_seq = torch.zeros(sim_len)
        
        # Looping over events in original sequence
        for idx2 in range(event_times_seq.shape.__getitem__(-1)):
            # Identify which simulated event times occur after the idx2th event time
            duration_index = sim_times_seq > event_times_seq[idx2].item()
            
            # If condition is true, subtract original event time from simulated, eventually gives time from most recent event to that time point
            sim_diffs_seq[duration_index] = sim_times_seq[duration_index] - event_times_seq[idx2]
            
            # Saves most recent event in orginal sequence that happens before simulated time
            sim_diffs_index[idx][duration_index] = idx2
        
        # Saves difference tensor
        sim_diffs_seqs[idx, :] = sim_diffs_seq[:]
    
    total_duration_seqs = torch.tensor(total_duration_seqs)
    return sim_diffs_seqs, sim_diffs_index, total_duration_seqs


class Data_Batch:
    def __init__(self, interarrival_seqs, event_types_seqs, length_seqs):
        # Saves process information to the class
        self.interarrival_seqs = interarrival_seqs
        self.event_types_seqs = event_types_seqs
        self.length_seqs = length_seqs

    def __len__(self):
        # Finds max_len of batch
        return self.event_types_seqs.shape[0]

    def __getitem__(self, index):
        # Name of tensors in batch
        batch = {
            'interarrival_seqs': self.interarrival_seqs[index],
            'event_types_seqs': self.event_types_seqs[index],
            'length_seqs': self.length_seqs[index]
        }
        return batch


if __name__ == "__main__":
    print(hi)