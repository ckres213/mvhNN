import argparse
import utils
import sys
import os
import datetime
import time
from torch.utils.data import DataLoader
import conttime
import torch
import matplotlib.pyplot as plt
import pdb
import re
import torch.nn.functional as F

#adding a comment before pushing, delete this line when read.
__author__ = 'Matt Pyper'
#log_valid(dev_interarrival_seqs, dev_event_types_seqs, dev_length_seqs, device)

def log_valid(interarrival_seqs, event_types_seqs, length_seqs, device):
    model = torch.load(f"./models/{dataset}.pt")
    length_seqs = torch.tensor(length_seqs)
    
    # Creates simulated sequences from the dev data, and saves the total sequence duration of the original dev sequences
    sim_diffs_seqs, sim_diffs_index, total_duration_seqs = utils.generate_simulation(interarrival_seqs, length_seqs)
    
    interarrival_seqs.to(device)
    event_types_seqs.to(device)
    length_seqs.to(device)
    
    sim_diffs_seqs.to(device)
    total_duration_seqs.to(device)    
    sim_diffs_index.to(device)
    
    # Calculate hidden states and gates using dev data
    hidden_states, cell_gates, cell_bar_gates, decay_gates, output_gates = model.forward(event_types_seqs, interarrival_seqs)
    
    # Calculate log-likelihood parts
    part_one_likelihood, part_two_likelihood, sum_likelihood = model.conttime_loss(hidden_states, 
                                                                                   cell_gates, 
                                                                                   cell_bar_gates, 
                                                                                   decay_gates, 
                                                                                   output_gates, 
                                                                                   event_types_seqs, 
                                                                                   sim_diffs_seqs, 
                                                                                   total_duration_seqs, 
                                                                                   length_seqs, 
                                                                                   sim_diffs_index
                                                                                  )
                                                                                   
    
    total_size = torch.sum(length_seqs)
   
    #cb
    log_likelihood = torch.sum(part_one_likelihood - part_two_likelihood) / total_size
    
    type_likelihood = torch.sum(part_one_likelihood - sum_likelihood) / total_size
    
    time_likelihood = log_likelihood - type_likelihood
    
    return log_likelihood, type_likelihood, time_likelihood

def type_valid(interarrival_seqs, length_seqs, event_types_seqs):
    model = torch.load(f"./models/{dataset}.pt")
    
    # Number of dev sequences
    num_dev_seqs = interarrival_seqs.shape[0]
   
    original_event_types = []
    predicted_event_types = []
    
    # Looping over the number of dev sequences
    for i in range(num_dev_seqs):
        # Extracting interarrival times, event types and length of the i^th sequence
        pdb.set_trace()
        interarrival_seq = interarrival_seqs[i:i+1]
        event_types_seq = event_types_seqs[i:i+1]
        length_seq = length_seqs[i]
        
        # Extracts final event type in sequence
        original_event_types.append(event_types_seq[0][length_seq].item())
        
        # 'unpads' the sequence
        event_types_seq = event_types_seq[:, :length_seq]
        interarrival_seq = interarrival_seq[:, :length_seq+1]
        
        # Calculates hidden state from sequence (forward)
        hidden_states, *_ = model.forward(event_types_seq, interarrival_seq)
        
        # Calculates intensity at final event? #cb
        lambda_all = F.softplus(model.hidden_lambda(hidden_states[-1]))
        lambda_sum = torch.sum(lambda_all, dim=-1)
        
        # Probability????
        lambda_all = lambda_all / lambda_sum
        
        # Wount predicted type be a random choice like in generate sequences?
        _, predict_type = torch.max(lambda_all, dim=-1)
        
        predicted_event_types.append(predict_type.item())
    
    # Counter for number of correctly predicted event types for the last event
    num_correct = 0
    
    for idx in range(num_dev_seqs):
        if predicted_event_types[idx] == original_event_types[idx]:
            num_correct += 1
    
    return num_correct / num_dev_seqs


if __name__ == "__main__":
    # User inputs information to train model 
    parser = argparse.ArgumentParser(description="Training model..")

    parser.add_argument(
        "-d", "--Dataset", 
        type=str, 
        help="e.g. conttime_x_Seq_x_Dim_x_MaxLen_x_LSTMUnits", 
        required=True
    )
    
    parser.add_argument(
        "-lr", "--LearningRate", 
        type=float, 
        default=0.01, 
        help="learning rate of " #cb
    )
    
    parser.add_argument(
        "-e", "--Epochs", 
        type=int, 
        default=10, 
        help="maximum epochs"
    )
    
    parser.add_argument(
        "--SeqLen", 
        type=int, 
        default=-1, 
        help="truncated sequence length for hawkes and self-correcting, -1 means full sequence"
    ) #cb
    
    parser.add_argument(
        "-bs", "--BatchSize", 
        type=int, 
        default=10, 
        help="Batch_size for each train iteration"
    )
    
    parser.add_argument(
        "-fp", "--FilePretrain", 
        type=bool, 
        default = False,
        help="True to use a trained model named model.pt"
    )

    args = parser.parse_args()
    
    # Saving arguments to variables
    dataset = args.Dataset
    learn_rate = args.LearningRate
    seq_len = args.SeqLen
    num_epochs = args.Epochs
    batch_size = args.BatchSize
    pretrain_model = args.FilePretrain

    # Creating a log file
    current_time = datetime.datetime.now().strftime("%d-%m-%H:%M")

    log_file_name = f"./train_output/{dataset}_{current_time}.txt"
    log = open(log_file_name, 'w')
    log.write("\nTraining data: " + dataset)
    log.write("\nLearning rate: " + str(learn_rate))
    log.write("\nMax epochs: " + str(num_epochs))
    log.write("\nSequence lengths: " + str(seq_len))
    log.write("\nBatch size for train: " + str(batch_size))
    log.write("\nPrevious model: " + str(pretrain_model))

    t1 = time.time()
    
    # Loading and processing data based on the specified dataset
    print("Processing data...")
    
    '''
    if dataset == 'hawkes' or dataset == "self-correcting":
        file_path = 'data/' + dataset + "/time-train.txt"   # train file
        test_path = 'data/' + dataset + '/time-test.txt'    # test file
        interarrival_seqs, length_seqs = utils.open_txt_file(file_path)   # train time info
        dev_interarrival_seqs, dev_length_seqs = utils.open_txt_file(test_path)   # test time info
        dim_process = 1
        event_types_seqs = utils.get_index_txt(interarrival_seqs) # train type
        dev_event_types_seqs = utils.get_index_txt(dev_interarrival_seqs)  # test type
        if seq_len == -1:
            interarrival_seqs, event_types_seqs = utils.padding_full(interarrival_seqs, event_types_seqs, length_seqs, dim_process)
            dev_interarrival_seqs, dev_event_types_seqs = utils.padding_full(dev_interarrival_seqs, dev_event_types_seqs, dev_length_seqs, dim_process)
        else:
            interarrival_seqs, event_types_seqs, length_seqs = utils.padding_seq_len(interarrival_seqs, event_types_seqs, dim_process, seq_len)
            dev_interarrival_seqs, dev_event_types_seqs = utils.padding_seq_len(dev_interarrival_seqs, dev_event_types_seqs, dim_process, seq_len)
    else:
        if dataset == 'conttime' or dataset == "data_hawkes" or dataset == "data_hawkeshib":
            dim_process = 8
        elif dataset == 'data_mimic1' or dataset == 'data_mimic2' or dataset == 'data_mimic3' or dataset == 'data_mimic4' or\
        dataset == 'data_mimic5':
            dim_process = 75
        elif dataset == 'data_so1' or dataset == 'data_so2' or dataset == 'data_so3' or dataset == 'data_so4' or\
        dataset == 'data_so5':
            dim_process = 22
        elif dataset == 'data_book1' or dataset == 'data_book2' or dataset == 'data_book3' or dataset == 'data_book4'\
        or dataset == 'data_book5':
            dim_process = 3
        else:
            print("Data process file for other types of datasets have not been developed yet, or the dataset is not found")
            log.write("\nData process file for other types of datasets have not been developed yet, or the datase is not found")
            log.close()
            sys.exit()
     '''
    # Loading and padding train and dev data
    data_path = f'data/{dataset}.pkl'
    
    # Extracting dim_process from dataset
    match = re.search(re.compile(r'_(\d+)_Dim_'), dataset)
    dim_process = int(match.group(1))
    
    # Extracting lstm_units from dataset
    match = re.search(re.compile(r'_(\d+)_LSTMUnits'), dataset)
    lstm_units = int(match.group(1))
    
    # Training data (seqs is sequences)
    # Extracts interarrival times, event types and total number of events for each sequence and saves in seperate lists. 
    interarrival_seqs, event_types_seqs, length_seqs = utils.open_pkl_file(data_path, 'train')
    
    # Pads the interarrival and event lists so they are all the same length. Makes first event for each tensor the BOS event. Makes first value of each interarrival tensor 0 
    interarrival_seqs, event_types_seqs = utils.padding_full(
        interarrival_seqs, 
        event_types_seqs, 
        length_seqs, 
        dim_process
    )
    
    # Same as above for validation data
    dev_interarrival_seqs, dev_event_types_seqs, dev_length_seqs = utils.open_pkl_file(data_path, 'dev')
    dev_interarrival_seqs, dev_event_types_seqs = utils.padding_full(
        dev_interarrival_seqs, 
        dev_event_types_seqs, 
        dev_length_seqs, 
        dim_process
    )

    # Saves data in class to be entered into dataloader
    train_data = utils.Data_Batch(interarrival_seqs, event_types_seqs, length_seqs)
    # Puts data in batches and shuffles to avoid overfitting
    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    print("Data Processing Finished...")
    
    t2 = time.time()
    data_process_time = t2 - t1
    print("Getting data takes: " + str(data_process_time) + " seconds")
    log.write("\n\nGetting data takes: " + str(data_process_time) + " seconds")

    
    # Train on data
    print("start training...")
    t3 = time.time()
    
    
    if pretrain_model:
        # Specified pretrained model 
        model = torch.load(f"./models/{dataset}.pt")
    else:
        # Initalizes model. Creates embedding layer, lstm_cell, hidden_lamda and the optimizer
        model = conttime.Conttime(dim_process = dim_process, lstm_units = lstm_units, learn_rate = learn_rate)

    # States what device train_final is running on    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("You are using GPU acceleration.")
        log.write("\nYou are using GPU acceleration.")
        print("Number of GPU: ", torch.get_num_threads())
        log.write("\n\nNumber of GPU: " + str((torch.get_num_threads())))
    else:
        device = torch.device("cpu")
        print("CUDA is not Available. You are using CPU only.")
        log.write("\nCUDA is not Available. You are using CPU only.")
        print("Number of cores: ", os.cpu_count())
        log.write("\n\nNumber of cores: " + str(os.cpu_count()))
    
    #cb make sure these are correct
    # Lists to store values for monitoring during training
    loss_value = []          # List to store negative log-likelihood values during training
    log_test_list = []       # List to store negative log-likelihood values during validation on test data
    log_time_list = []       # List to store negative log-likelihood values for time during validation on test data
    log_type_list = []       # List to store negative log-likelihood values for event type during validation on test data
    type_accuracy_list = []  # List to store accuracy values for event type prediction during validation on test data

    # Loop through epochs for training
    for i in range(num_epochs):
        total_loss = 0
        total_events = 0 #cb
        
        max_len = len(train_data)
        
        for idx, batch in enumerate(train_data):    
            
            # Save batch information as individual tensors
            interarrival_batch, event_types_batch, length_seqs_batch = batch['interarrival_seqs'], batch['event_types_seqs'], batch['length_seqs']
            
            # Generate simulation
            sim_diffs_seqs, sim_diffs_index, total_duration_seqs = utils.generate_simulation(interarrival_batch, length_seqs_batch)
            
            event_types_batch.to(device)
            interarrival_batch.to(device)
            sim_diffs_seqs.to(device)
            total_duration_seqs.to(device)
            length_seqs_batch.to(device)
            sim_diffs_index.to(device)
            
            # Trains model and outputs loss
            loss = model.train_batch(interarrival_batch, event_types_batch, sim_diffs_seqs, total_duration_seqs, length_seqs_batch, sim_diffs_index)
            
            
            log_likelihood = -loss
            # Adds log likelihood of the batch to the total log likelihood count 
            total_loss += log_likelihood.item() #cb, loss and log-likelihood seem too interchangeable. 
            # Adds number of events in batch to the total event count
            total_events += torch.sum(length_seqs_batch).item()
            
            print("In epochs {0}, process {1} over {2} is done".format(i, idx, max_len))
            
        
        # Average "loss" per event
        avg_log = total_loss / total_events
        
        loss_value.append(-avg_log)
        
        print(f"The log-likelihood per event at epochs {i} is {avg_log}")
        log.write(f"\nThe log likelihood per event at epochs {i} is {avg_log}")
        print("model saved..")a
        
        torch.save(model, f"./models/{dataset}.pt")

        print("\nvalidating on log likelihood...")
        log_likelihood, type_likelihood, time_likelihood = log_valid(dev_interarrival_seqs, dev_event_types_seqs, dev_length_seqs, device)
        log_test_list.append(-log_likelihood.item())
        log_type_list.append(-type_likelihood.item())
        log_time_list.append(-time_likelihood.item())

        print("\nvalidating on type prediction accuracy if we know when will next event happens...\n\n")
        accuracy = type_valid(dev_interarrival_seqs, dev_length_seqs, dev_event_types_seqs)
        type_accuracy_list.append(accuracy)

    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
    figure.suptitle(f"{dataset}'s Training Figure {current_time}")
    ax[0].set_xlabel("epochs")
    ax[0].plot(loss_value, label='training loss')
    ax[0].plot(log_test_list, label='testing loss')
    ax[0].legend()
    ax[1].set_xlabel("epochs")
    ax[1].plot(log_type_list, label='testing type loss')
    ax[1].plot(log_time_list, label='testing time loss')
    ax[1].legend()
    ax[2].set_xlabel("epochs")
    ax[2].set_ylabel('accuracy')
    ax[2].set_title('type-validation-accuracy')
    ax[2].plot(type_accuracy_list, label='dev type accuracy')
    plt.subplots_adjust(top=0.85)
    figure.tight_layout()
    plt.savefig(f"./train_output/{dataset}_train_{current_time}.jpg")

    t4 = time.time()
    training_time = t4 - t3
    print("training done..")
    print("training takes {0} seconds".format(training_time))
    log.write("\ntraining takes {0} seconds".format(training_time))
    log.close()

    print("Saving training loss and validation data...")
    print("If you have a trained model before this, please combine the previous train_date file to" +
        " generate plots that are able to show the whole training information")
    training_info_file = f"./train_output/{dataset}_train_{current_time}.txt"
    file = open(training_info_file, 'w')
    file.write("log-likelihood: ")
    file.writelines(str(item) + " " for item in loss_value)
    file.write('\nlog-test-likelihood: ')
    file.writelines(str(item) + " " for item in log_test_list)
    file.write('\nlog-type-likelihood: ')
    file.writelines(str(item) + " " for item in log_type_list)
    file.write('\nlog-time-likelihood: ')
    file.writelines(str(item) + " " for item in log_time_list)
    file.write('\naccuracy: ')
    file.writelines(str(item) + " " for item in type_accuracy_list)
    file.close()
    print("Every works are done!")


