# -*- coding: utf-8 -*-
"""

Here are the sequence generators
including LSTM generator and Hawkes generator

@author: Matt Pyper
"""

import pickle
import time
import numpy as np
import os
from collections import defaultdict
import struct
import pdb

dtype=np.float64
 
class HawkesGen(object):
    def __init__(self, generator_settings):       
        """
        Initialize the Hawkes sequence generator object.
        """
        
        self.args = generator_settings['args']
        self.sum_for_time = generator_settings['sum_for_time']
        np.random.seed(generator_settings['seed'])
        
        
        if generator_settings['pretrain_model_path'] == None:
            # Randomly selects parameter values (stationarity guranteed as alpha
            
            self.dim_process = generator_settings['dim_process']
            
            self.mu = np.float32(
                np.random.uniform(low=0.0, high=1.0,size=(self.dim_process,))
            )
            
            self.alpha = np.float32(
                np.random.uniform(low=0.0, high=1.0, size=(self.dim_process, self.dim_process))
            )
            
            self.delta = np.float32(
                np.random.uniform(low=10.0, high=20.0, size=(self.dim_process, self.dim_process))
            )
            
        else:
            pretrain_model_path = os.path.abspath(generator_settings['pretrain_model_path'])
            
            with open(pretrain_model_path, 'rb') as f:
                pretrain_model = pickle.load(f)
            
            self.dim_process = pretrain_model['dim_process']
            self.mu = pretrain_model['mu']
            self.alpha = pretrain_model['alpha']
            self.delta = pretrain_model['delta']
               
        
        self.name = 'HawkesGen'
        
        # Initalizing intensity (background rate)
        self.intensity = np.copy(self.mu)
        
        # Empty list for one sequence and counter for number of events in sequence
        self.one_sequence = []
        self.cnt_total_event = np.int32(len(self.one_sequence))
        
       
    def set_args(self, dict_args):
        # Saves model information to class 
        self.args = dict_args

        
    def save_model(self, file_save):
        print( "saving model of generator ... " )
        model_dict = {
            'mu': np.copy(self.mu),
            'alpha': np.copy(self.alpha),
            'delta': np.copy(self.delta),
            'dim_process': self.dim_process,
            'name': self.name,
            'args': self.args
        }
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
    
    
    def restart_sequence(self):
        # Reset intensity and list sequence is stored
        self.intensity = np.copy(self.mu)
      
        self.one_sequence = []
        self.cnt_total_event = np.int32(len(self.one_sequence))

        
    def compute_intensity_given_past(self, time_current):
        # Initialize intensity with background rate (already is not SFT)
        self.intensity = np.copy(self.mu)
        
        # Looping over previous events
        for event in self.one_sequence:
            # Calculate intensity form equation (2)
            time_since_start = event['time_since_start']
            event_type = event['event_type']
            change_time = time_current - time_since_start
            
            decay_frac = np.exp(-self.delta[:, event_type] * change_time)

            self.intensity += np.copy(self.alpha[:, event_type] * decay_frac)
        
        # No upper bound used #cb
    def sample_time_given_type(self, event_type):
        """
        Sample time for a given event type in the Hawkes process.

        Parameters:
        - event_type (int): The event type for which to sample the time.
        """
        
        # Set time to 0
        time_current = np.float32(0.0)
        
        # If previous events in sequence have been sampled, take most recent time
        if len(self.one_sequence) > 0:
            time_current = self.one_sequence[-1]['time_since_start']
        
        # Compute intensity for each event type at the current time
        self.compute_intensity_given_past(time_current)
        intensity_hazard = np.copy(self.intensity[event_type])
        
        u = 1.5
        
        while u >= 1.0:
            # Select random value from Exp(1) distribution 
            E = np.random.exponential(scale=1.0, size=None) 

            # Select random value from Unif(0,1) distribution
            U = np.random.uniform(low=0.0, high=1.0, size=None) 

            # Increase time by small increment
            time_current += E / intensity_hazard

            # Recompute intensity given increment in time
            self.compute_intensity_given_past(time_current)

            # Recompute u 
            u = U * intensity_hazard / self.intensity[event_type]

            # "Adaptive thinning", decreases upper bound
            intensity_hazard = np.copy(self.intensity[event_type])
        return time_current

    
    def sample_time_for_all_type(self):
        """
        Sample next event time using a Hawkes process.
        """
        # Set time to 0
        time_current = np.float32(0.0)
        
        # If previous events in sequence have been sampled, take most recent time
        if len(self.one_sequence) > 0:
            time_current = self.one_sequence[-1]['time_since_start']
        
        # Compute intensity for each event type at the current time 
        self.compute_intensity_given_past(time_current)
        
        # Compute total intensity across all event types
        intensity_hazard = np.sum(self.intensity)
      
        u = 1.5
        
        while u >= 1.0:
            # Selects random value from Exp(1) distribution 
            E = numpy.random.exponential(
                scale=1.0, size=None
            ) 
            
            # Selects random value from Unif(0,1) distribution
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            ) 
            
            # Increases time by small increment
            time_current += E / intensity_hazard
            # Recomputes intensity given increment in time
            self.compute_intensity_given_past(time_current)
            
            # Recompute u 
            u = U * intensity_hazard / np.sum(self.intensity)
           
            # "Adaptive thinning", decreases upper bound
            intensity_hazard = np.sum(self.intensity)
        return time_current

    
    def sample_one_event_sep(self):
        # Time of next event for each event type (1xn array)
        time_of_happen = np.zeros((self.dim_process,), dtype=dtype)
        
        # Looping over each event type in the process
        for event_type in range(self.dim_process):
            
            # Sample one event with event type event_type using "thinning algorithm"
            time_of_happen[event_type] = np.copy(
                self.sample_time_given_type(event_type)
            )
        
        # Chooses event that happens first
        time_since_start_new = np.min(time_of_happen)
        event_type_new = np.argmin(time_of_happen)
        return time_since_start_new, event_type_new

    
    def sample_one_event_tog(self):
        # Finds when next event happens using total intensity across event types
        time_since_start_new = self.sample_time_for_all_type()
        
        # Compute intensity given new time
        self.compute_intensity_given_past(time_since_start_new)
        
        # Transforms intensity of each event type to a probability
        prob = self.intensity / np.sum(self.intensity)
        
        # Choose event type based off those probabilities        
        event_type_new = np.random.choice(
            range(self.dim_process), p = prob
        )
        return time_since_start_new, np.int32(event_type_new)

    
    def sample_one_event(self):
        # Samples events using total intensity or event intensity
        if self.sum_for_time:
            return self.sample_one_event_tog()
        else:
            return self.sample_one_event_sep()

        
    def gen_one_sequence(self, max_len):
        """
        Generates a single sequence of a Hawkes process.

        Parameters:
        - max_len (int): Pre-sampled value to set the maximum length of the sequence.

        This method generates a single sequence of the Hawkes process using a thinning algorithm.
        It iteratively samples events, updates the sequence, and records relevant information about each event.
        The maximum length of the sequence is controlled by the pre-sampled value 'max_len'.
        """
        
        self.restart_sequence()
        
        # Initialize time overall and for each event type
        time_since_start = np.float32(0.0)
        time_since_start_each_event = np.zeros(
            (self.dim_process,), dtype=dtype
        )
        
        # Looping over each event in the sequence:
        for event_idx in range(max_len):
            time_since_start_new, event_type_new = self.sample_one_event()
            self.cnt_total_event += 1
            
            # Update sequence
            time_since_last_event = time_since_start_new - time_since_start
            time_since_start = time_since_start_new
            time_since_last_same_event = time_since_start -time_since_start_each_event[event_type_new]
            time_since_start_each_event[event_type_new] = time_since_start
            
            self.one_sequence.append({'event_idx': self.cnt_total_event,
                                         'event_type': event_type_new,
                                         'time_since_start': time_since_start,
                                         'time_since_last_event': time_since_last_event,
                                         'time_since_last_same_event': time_since_last_same_event})

            
    def gen_sequences(self, sequence_settings):
        """
        Generates multiple sequences of a Hawkes process.

        Parameters:
        - sequence_settings (dict): A dictionary containing generation settings.
            - num_sequences: Number of sequences to generate.
            - min_len: Minimum length of each sequence.
            - max_len: Maximum length of each sequence.

        The method generates 'num_sequences' sequences of the Hawkes process, each with a random length
        between 'min_len' and 'max_len'.
        """
     
        num_sequences = sequence_settings['num_sequences']
        
        # List of generated sequences
        self.list_sequences = []
        # Current sequence to be generated
        sequence_cnt = 0
        
        while sequence_cnt < num_sequences:
            # Randomly select length of sequence 
            max_len = np.int32(
                round(np.random.uniform(low=sequence_settings['min_len'], high=sequence_settings['max_len']))
            )
            
            # Generate one sequence
            self.gen_one_sequence(max_len)
            # Save sequence
            self.list_sequences.append(self.one_sequence)
            sequence_cnt += 1
            
            # Print progress to user
            if sequence_cnt % 10 == 0:
                print( "idx sequence of gen : ", (sequence_cnt, self.name) )
                print( "total number of sequences : ", num_sequences )

                
    def save_sequences(self, file_save):
        with open(file_save, 'wb') as f:
            pickle.dump(self.list_sequences, f)


class HawkesInhibGen(object):
    def __init__(self, generator_settings):
        """
        Initialize the Hawkes Inhibition sequence generator.
        """
        
        self.args = generator_settings['args']
        self.sum_for_time = generator_settings['sum_for_time']
        np.random.seed(generator_settings['seed'])
        
        if generator_settings['pretrain_model_path'] == None:
        # Randomly selects parameter values (stationarity guranteed as alpha/delta < 1
            self.dim_process = generator_settings['dim_process']
            
            self.mu = np.float32(
                np.random.uniform(low = -1.0, 
                                  high = 1.0, 
                                  size = (self.dim_process,)
                                 )
            )
            
            self.alpha = np.float32(
                np.random.uniform(low = -1.0, 
                                  high = 1.0, 
                                  size = (self.dim_process, self.dim_process)
                                 )
            )
            
            self.delta = np.float32(
                np.random.uniform(low=10.0, 
                                  high=20.0, 
                                  size=(self.dim_process, self.dim_process)
                                 )
            )
            
        else: # Previous model specified
            pretrain_model_path = os.path.abspath(generator_settings['pretrain_model_path'])
            
            with open(pretrain_model_path, 'rb') as f:
                pretrain_model = pickle.load(f)
            
            self.dim_process = pretrain_model['dim_process']
            self.mu = pretrain_model['mu']
            self.alpha = pretrain_model['alpha']
            self.delta = pretrain_model['delta']
            

        self.name = 'HawkesInhibGen'
        
        # Initalizing intensity (background rate)
        self.intensity_tilde = np.copy(self.mu)
        
        # Softplus to make intensity positive (In full code use torch function) #cb
        self.intensity = np.log(np.float32(1.0) + np.exp(self.intensity_tilde))
        
        # Upper bounds for intensities
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        
        self.one_sequence = []
        self.cnt_total_event = np.int32(len(self.one_sequence))

        
    def set_args(self, dict_args):
        # Saves model information to class 
        self.args = dict_args

    def soft_relu(self, x):
        # Softplus function 
        return np.log(np.float32(1.0)+np.exp(x))
    
    """ No scale used in hawkes inbib #cb 
    def soft_relu_scale(self, x):
        # last dim of x is dim_process
        x /= self.scale
        y = np.log(np.float32(1.0)+np.exp(x))
        y *= self.scale
        return y
    """
    def hard_relu(self, x):
        # ReLU function
        return np.float32(0.5) * (x + np.abs(x) )

    
    def save_model(self, file_save):
        print( "saving model of generator ... " )
        
        model_dict = {'mu': np.copy(self.mu),
                      'alpha': np.copy(self.alpha),
                      'delta': np.copy(self.delta),
                      'dim_process': self.dim_process,
                      'name': self.name,
                      'args': self.args
                     }
        
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
 

    def restart_sequence(self):
        # Reset intensities and list sequence is stored
        self.intensity_tilde = np.copy(self.mu)
        self.intensity = self.soft_relu(self.intensity_tilde)
   
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        
        self.one_sequence = []
        self.cnt_total_event = np.int32(len(self.one_sequence))

        
    def compute_intensity_given_past(self, time_current):
        # Initialize intensity with background rate
        self.intensity_tilde = np.copy(self.mu)
        
        # Looping over previous events
        for event in self.one_sequence:
            # Calculate intensity form equation (2)
            time_since_start = event['time_since_start']
            event_type = event['event_type']
            change_time = time_current - time_since_start
            
            decay_frac = np.exp(-self.delta[:, event_type] * change_time)

            self.intensity += np.copy(self.alpha[:, event_type] * decay_frac)
        
        self.intensity = self.soft_relu(self.intensity_tilde)

    
    def compute_intensity_upper_bound(self, time_current):
        # Initialize upper bound with background rate
        self.intensity_tilde_ub = np.copy(
            self.mu
        )
        
        # Looping over previous events
        for event in self.one_sequence:            
            change_time = time_current - event['time_since_start']
            
            decay_frac = np.exp(
                -self.delta[:, event['event_type']] * change_time
            )
            
            # Will set any negative alpha value to 0 (keeping upper bound)
            self.intensity_tilde_ub += np.copy(
                self.hard_relu(self.alpha[:, event['event_type']]) * decay_frac
            )
        
        self.intensity_ub = self.soft_relu(self.intensity_tilde_ub)

        
    def sample_time_given_type(self, event_type):
        """
        Sample next event time for a given event type using a Hawkes process with inhibition.

        Parameters:
        - event_type (int): The event type for which to sample the time.
        """
        # Set time to 0
        time_current = np.float32(0.0)
        
        # If previous events in sequence have been sampled, take most recent time
        if len(self.one_sequence) > 0:
            time_current = self.one_sequence[-1]['time_since_start']
        
        # Compute upper bound of intensity at the current time
        self.compute_intensity_upper_bound(time_current)
        
        intensity_hazard = np.copy(self.intensity_ub[event_type])
        
        u = 1.5
        
        while u >= 1.0:
            # Select random value from Exp(1) distribution 
            E = np.random.exponential(scale=1.0, size=None) 

            # Select random value from Unif(0,1) distribution
            U = np.random.uniform(low=0.0, high=1.0, size=None) 

            # Increase time by small increment
            time_current += E / intensity_hazard

            # Recompute intensity given increment in time
            self.compute_intensity_given_past(time_current)

            # Recompute u 
            u = U * intensity_hazard / self.intensity[event_type]

            # "Adaptive thinning", decreases upper bound          
            self.compute_intensity_upper_bound(time_current)
            intensity_hazard = np.copy(self.intensity_ub[event_type])
        
        return time_current

    
    def sample_time_for_all_type(self):
        """
        Sample next event time using a Hawkes process with inhibition.
        """
        # Set time to 0
        time_current = numpy.float32(0.0)
        
        # If previous events in sequence have been sampled, take most recent time
        if len(self.one_sequence) > 0:
            time_current = self.one_sequence[-1]['time_since_start']
        
        # Compute upper bound of intensity at the current time 
        self.compute_intensity_upper_bound(time_current)
        
        # Total upper bound of intensity
        intensity_hazard = numpy.sum(self.intensity_ub)
        
        u = 1.5
        
        while u >= 1.0:
            # Selects random value from Exp(1) distribution 
            E = numpy.random.exponential(
                scale=1.0, size=None
            ) 
            
            # Selects random value from Unif(0,1) distribution
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            ) 
            
            # Increases time by small increment
            time_current += E / intensity_hazard
            # Recomputes intensity given increment in time
            self.compute_intensity_given_past(time_current)
            
            # Recomputes u 
            u = U * intensity_hazard / np.sum(self.intensity)
            
            # "Adaptive thinning", decreases upper bound            
            self.compute_intensity_upper_bound(time_current)
            intensity_hazard = np.sum(self.intensity_ub)
         
        return time_current

    
    def sample_one_event_sep(self):
        # Time of next event for each event type
        time_of_happen = np.zeros(
            (self.dim_process,), dtype=dtype
        )
        
        # Looping over each event type in the process
        for event_type in range(self.dim_process):
            # Sample one event with event type event_type using "thinning algorithm"
            time_of_happen[event_type] = np.copy(
                self.sample_time_given_type(event_type)
            )
            
        # Choose event that happens first
        time_since_start_new = np.min(time_of_happen)
        event_type_new = np.argmin(time_of_happen)
        
        return time_since_start_new, event_type_new

    
    def sample_one_event_tog(self):
        # Finds when next event happens using total intensity across event types
        time_since_start_new = self.sample_time_for_all_type()
        
        # Compute intensity given new time
        self.compute_intensity_given_past(time_since_start_new)
        
        # Transforms intensity of each event type to a probability
        prob = self.intensity / np.sum(self.intensity)
        
        #  Choose event type based off those probabilities 
        event_type_new = np.random.choice(range(self.dim_process), p = prob)
        
        return time_since_start_new, np.int32(event_type_new)

    
    def sample_one_event(self):
        # Samples events using total intensity or event intensity
        if self.sum_for_time:
            return self.sample_one_event_tog()
        else:
            return self.sample_one_event_sep()

        
    def gen_one_sequence(self, max_len):
        """
        Generates a single sequence of a Hawkes process with inhibition.

        Parameters:
        - max_len (int): Pre-sampled value to set the maximum length of the sequence.

        This method generates a single sequence of the Hawkes process using a thinning algorithm.
        It iteratively samples events, updates the sequence, and records relevant information about each event.
        The maximum length of the sequence is controlled by the pre-sampled value 'max_len'.
        """
        
        self.restart_sequence()
        
        # Initialize time overall and for each event type
        time_since_start = np.float32(0.0)
        
        time_since_start_each_event = np.zeros(
            (self.dim_process,), dtype=dtype
        )       
        
        # Looping over each event in the sequence:
        for event_idx in range(max_len):
            time_since_start_new, event_type_new = self.sample_one_event()
            self.cnt_total_event += 1
            
            # Update sequence
            time_since_last_event = time_since_start_new - time_since_start
            time_since_start = time_since_start_new
            time_since_last_same_event = time_since_start - time_since_start_each_event[event_type_new]
            time_since_start_each_event[event_type_new] = time_since_start
            
            self.one_sequence.append({'event_idx': self.cnt_total_event,
                                         'event_type': event_type_new,
                                         'time_since_start': time_since_start,
                                         'time_since_last_event': time_since_last_event,
                                         'time_since_last_same_event': time_since_last_same_event})

            
    def gen_sequences(self, sequence_settings):
        """
        Generate multiple sequences of the Hawkes process.

        Parameters:
        - sequence_settings (dict): A dictionary containing generation settings.
            - sequences: Number of sequences to generate.
            - min_len: Minimum length of each sequence.
            - max_len: Maximum length of each sequence.

        The method generates 'num_sequences' sequences of the Hawkes process, each with a random length
        between 'min_len' and 'max_len'.
        """
     
        num_sequences = sequence_settings['num_sequences']
        
        # List of generated sequences
        self.list_sequences = []
        # Current sequence to be generated
        sequence_cnt = 0
        
        while sequence_cnt < num_sequences:
            # Randomly select length of sequence 
            max_len = np.int32(
                round(np.random.uniform(low=sequence_settings['min_len'], high=sequence_settings['max_len']))
            )
            
            # Generate one sequence
            self.gen_one_sequence(max_len)
            # Save sequence
            self.list_sequences.append(self.one_sequence)
            sequence_cnt += 1
            
            # Print progress to user
            if sequence_cnt % 10 == 0:
                print( "idx sequence of gen : ", (sequence_cnt, self.name) )
                print( "total number of sequences : ", num_sequences )

                
    def save_sequences(self, file_save):
        with open(file_save, 'wb') as f:
            pickle.dump(self.list_sequences, f)


class NeuralHawkesCTLSTM(object):
    def __init__(self, generator_settings):
        """
        Initialize the Neural Hawkes CTLSTM sequence generator.
        """
        
        self.args = generator_settings['args']
        self.sum_for_time = generator_settings['sum_for_time']
        
        # If previous model not specified, select parameters randomly
        if generator_settings['pretrain_model_path'] == None:
            self.dim_process = generator_settings['dim_process']
            self.lstm_units = generator_settings['lstm_units']
            
            np.random.seed(generator_settings['seed'])
            
            """
            s_k randomized here (commented out #cb)
            self.scale = np.float32(
                np.random.uniform(
                    low = 1e-3, high = 2.0,
                    size = (self.dim_process, )
                )
            )
            """
           
            self.scale = np.ones((self.dim_process, ))
            
            # The matrix of w^T_k from equation (4a)
            self.W_alpha = np.float32(
                np.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (self.lstm_units, self.dim_process)
                )
            )
            
             # The vector specifying the type of event, with the extra row for the first event
            self.Emb_event = np.float32(
                np.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (self.dim_process + np.int32(1), self.lstm_units)
                )
            )
            
            # W_{} and U_{} (weights) in equations (5a) through (5d), (6c), and to calculate \bar{f}, \bar{i}
            self.W_recur = np.float32(
                np.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (2 * self.lstm_units, 7 * self.lstm_units)
                )
            )
            
            # d_{} in equations (5a) through (5d), (6c), and to calculate \bar{f}, \bar{i}
            self.b_recur = np.float32(
                np.random.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (7 * self.lstm_units, )
                )
            )
        
        else: # If previous model is specified
            print( "read pretrained model ... " )
            
            # Save file path of pretrained model 
            pretrain_model_path = os.path.abspath(generator_settings['pretrain_model_path'])
            
            # Open pretrained model 
            with open(pretrain_model_path, 'rb') as f:
                pretrain_model = pickle.load(f)
            
            # Save pretrained models arguments 
            self.dim_process = pretrain_model['dim_process']
            self.lstm_units = pretrain_model['lstm_units']
            
            self.scale = pretrain_model['scale']
            self.W_alpha = pretrain_model['W_alpha']
            self.Emb_event = pretrain_model['Emb_event']
            self.W_recur = pretrain_model['W_recur']
            self.b_recur = pretrain_model['b_recur']
            
        self.name = 'NeuralHawkesGenCTLSTM'
        
        # Initializing intensity values and their upper bounds
        self.intensity_tilde = None
        self.intensity = None
        
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        
        self.one_sequence = []
        self.one_sequence.append({'event_idx': np.int32(0),
                                     'event_type': self.dim_process,
                                     'time_since_start': np.float32(0.0),
                                     'time_since_last_event': np.float32(0.0),
                                     'time_since_last_same_event': np.float32(0.0)})

        # Initialization of outputs of equations (5d) to (6c)
        # c_{i+1}
        self.cell = np.zeros((self.lstm_units, ), dtype = dtype)
        
        # bar{c}_{i+1}
        self.cell_bar = np.zeros((self.lstm_units, ), dtype = dtype)
        
        # \delta_{i+1}
        self.decay_gate = np.zeros((self.lstm_units, ), dtype = dtype)
        
        # o_i
        self.output_gate = np.zeros((self.lstm_units, ), dtype = dtype)
        
        self.cnt_total_event = np.int32(len(self.one_sequence) )

        
    def set_args(self, dict_args):
        # Saves model information to class
        self.args = dict_args
    
    
    def soft_relu(self, x):
        # Softplus function 
        return np.log(np.float32(1.0)+np.exp(x))
    
    
    def soft_relu_scale(self, x):
        # Softplus function with s_k
        x /= self.scale
        y = np.log(np.float32(1.0)+np.exp(x))
        y *= self.scale
        return y

    
    def hard_relu(self, x):
        # ReLU function
        return np.float32(0.5) * (x + np.abs(x) )

    
    def save_model(self, file_save):
        print( "saving model of generator ... " )
        
        model_dict = {'scale': np.copy(self.scale),
                      'W_alpha': np.copy(self.W_alpha),
                      'Emb_event': np.copy(self.Emb_event),
                      'W_recur': np.copy(self.W_recur),
                      'b_recur': np.copy(self.b_recur),
                      'dim_process': self.dim_process,
                      'lstm_units': self.lstm_units,
                      'name': self.name,
                      'args': self.args}
        
        with open(file_save, 'wb') as f:
            pickle.dump(model_dict, f)
    
    
    def restart_sequence(self):
        # clear the events memory and reset starting time is 0
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_sequence = []
        
        # initialization for LSTM states
        self.one_sequence.append({'event_idx': np.int32(0),
                                     'event_type': self.dim_process,
                                     'time_since_start': np.float32(0.0),
                                     'time_since_last_event': np.float32(0.0),
                                     'time_since_last_same_event': np.float32(0.0)})
        
        # Resetting gates
        self.cell = np.zeros((self.lstm_units, ), dtype = dtype)
        self.cell_bar = np.zeros((self.lstm_units, ), dtype = dtype)

        self.decay_gate = np.zeros((self.lstm_units, ), dtype = dtype)
        self.output_gate = np.zeros((self.lstm_units, ), dtype = dtype)

        self.cnt_total_event = np.int32(len(self.one_sequence) )

        
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    
    def compute_hidden_states(self):       
        interarrival = self.one_sequence[-1]['time_since_last_event']
        event_type = self.one_sequence[-1]['event_type']       

        # Equation (7)
        cell_t = (
            self.cell_bar +
            (self.cell - self.cell_bar) *
            np.exp(-self.decay_gate * interarrival)
        )

        
        hidden_t = self.output_gate * np.tanh(cell_t)
        
        #TODO: update
        # extracts event type
        emb_event_t = self.Emb_event[event_type, :]
        
        # One single calculation for all the W_k_ + U_h(t_i) + d_
        
        """
        input_i is a 1 x (2 * lstm_units) matrix. 
        W_recur is a 2 * self.lstm_units, 7 * self.lstm_units
        b_recur is a 1 x 2 * self.lstm_units
        
        output_i is a 1 x 2 * self.lstm_units array which is split up into 7 parts
        """
        # Inpu
        input_i = np.concatenate((emb_event_t, hidden_t), axis=0)
        output_i = np.dot(input_i, self.W_recur) + self.b_recur

       
        # Splitting the output_i tensor into \mathbf{i}_{i+1}, \mathbf{f}_{i+1}, \mathbf{z}_{i+1}, \mathbf{o}_{i+1} (5a), (5b), (5c), (5d) 
        input_gate = self.sigmoid(output_i[:self.lstm_units])
        forget_gate = self.sigmoid(output_i[self.lstm_units:2*self.lstm_units])
        output_gate = self.sigmoid(output_i[2*self.lstm_units:3*self.lstm_units])
        cell_input_gate = np.tanh(output_i[3*self.lstm_units:4*self.lstm_units])
        
        
        # Target input and forget gates \overline{\mathbf{i}}_{i+1}, \overline{\mathbf{f}}_{i+1}
        input_bar_gate = self.sigmoid(output_i[4*self.lstm_units:5*self.lstm_units])
        forget_bar_gate = self.sigmoid(output_i[5*self.lstm_units:6*self.lstm_units])
        
        
        # Cell memory decay \mathbf{delta}_{i+1} (6c)
        decay_gate = self.soft_relu(output_i[6*self.lstm_units:])
        
        
        # Cell and target cell values \mathbf{c}_{i+1}, \overline{\mathbf{c}}_{i+1} (6a) and (6b)
        cell = forget_gate * cell_t + input_gate * cell_input_gate
        cell_bar = forget_bar_gate * self.cell_bar + input_bar_gate * cell_input_gate
        
        
        # Copy the updated values to the class attributes
        self.cell = np.copy(cell)
        self.cell_bar = np.copy(cell_bar)
        self.decay_gate = np.copy(decay_gate)
        self.output_gate = np.copy(output_gate)
      
    
    def compute_intensity_given_past(self, time_current):
        # compute the intensity of current time
        
        # Most recent event time
        time_recent = self.one_sequence[-1]['time_since_start']
        
        # Equation (7)
        cell_t = (
            self.cell_bar +
            (self.cell - self.cell_bar) *
            np.exp(-self.decay_gate * (time_current - time_recent))
        )

        
        # Equation (4b)
        hidden_t = self.output_gate * np.tanh(cell_t)
        
        # \bold{w}_k^T \bold{h}(t)
        self.intensity_tilde = np.dot(hidden_t, self.W_alpha)
        
        # Equation (4a)
        self.intensity = self.soft_relu_scale(self.intensity_tilde)    
    
    def compute_intensity_upper_bound(self, time_current):
        # Vector of ones to calculate outer products with (outer product u ⊗ v = uv^T)
        ones = np.ones((self.dim_process, ), dtype=dtype)
            
        time_recent = self.one_sequence[-1]['time_since_start']
        
        # \mathbf{c}_{i+1} - \overline{\mathbf{c}}_{i+1}. exp(...) will not change sign of this matrix
        cell_gap = self.cell - self.cell_bar

        # Matrix where each column is cell_gap so that it conforms with W_alpha which is a lstm_units x dim_process matrix
        cell_gap_matrix = np.outer(cell_gap, ones)
        
        
        pdb.set_trace() #cb rename this 
        # Booleans to indicate index increasing 
        index_increasing_0 = (cell_gap_matrix > 0.0) & (self.W_alpha < 0.0)
        index_increasing_1 = (cell_gap_matrix < 0.0) & (self.W_alpha > 0.0)
        
        # Zeroes all elements that will cause element in intensity_tilde to be negative (ReLU)
        cell_gap_matrix[index_increasing_0] = np.float32(0.0)
        cell_gap_matrix[index_increasing_1] = np.float32(0.0)
        
        # Calculating hidden state across all event types        
        cell_t = (
            np.outer(self.cell_bar, ones) +
            cell_gap_matrix * np.exp(
                -np.outer(self.decay_gate, ones)
                ) * (time_current - time_recent)
        )
        
        
        hidden_t = np.outer(self.output_gate, 
                                        np.ones((self.dim_process, ), dtype=dtype
                                               )
                                       ) * np.tanh(cell_t)
              
        # Computing upper bound of intensity
        self.intensity_tilde_ub = np.sum(
            hidden_t * self.W_alpha, axis=0
        )
        
        self.intensity_ub = self.soft_relu_scale(
            self.intensity_tilde_ub
        )
        
    def sample_time_given_type(self, event_type):
        """
        Sample next event time for a given event type using a Neural Hawkes process.

        Parameters:
        - event_type (int): The event type for which to sample the time.
        """
        # Set time to 0
        time_current = np.float32(0.0)
        
        # If previous events in sequence have been sampled, take most recent time
        if len(self.one_sequence) > 0:
            time_current = self.one_sequence[-1]['time_since_start']
        
        # Compute upper bound of intensity at the current time
        self.compute_intensity_upper_bound(time_current)
        intensity_hazard = np.copy(self.intensity_ub[event_type])
        
        u = 1.5
        while u >= 1.0:
            # Select random value from Exp(1) distribution 
            E = np.random.exponential(scale=1.0, size=None) 

            # Select random value from Unif(0,1) distribution
            U = np.random.uniform(low=0.0, high=1.0, size=None) 

            # Increase time by small increment
            time_current += E / intensity_hazard

            # Recompute intensity given increment in time
            self.compute_intensity_given_past(time_current)

            # Recompute u 
            u = U * intensity_hazard / self.intensity[event_type]

            # "Adaptive thinning", decreases upper bound           
            self.compute_intensity_upper_bound(time_current)
            intensity_hazard = np.copy(self.intensity_ub[event_type])
        return time_current

    
    def sample_time_for_all_type(self):
        """
        Sample next event time using a Neural Hawkes process.
        """
        # Set time to 0
        time_current = np.float32(0.0)
        
        # If previous events in sequence have been sampled, take most recent time
        if len(self.one_sequence) > 0:
            time_current = self.one_sequence[-1]['time_since_start']
        
        # Compute upper bound of intensity at the current time 
        self.compute_intensity_upper_bound(time_current)
        
        # Total upper bound of intensity
        intensity_hazard = np.sum(self.intensity_ub)

        u = 1.5
        
        while u >= 1.0:
            # Selects random value from Exp(1) distribution 
            E = np.random.exponential(
                scale=1.0, size=None
            ) 
            
            # Selects random value from Unif(0,1) distribution
            U = np.random.uniform(
                low=0.0, high=1.0, size=None
            ) 
            
            # Increases time by small increment
            time_current += E / intensity_hazard
            # Recomputes intensity given increment in time
            self.compute_intensity_given_past(time_current)
            
            # Recomputes u 
            u = U * intensity_hazard / np.sum(self.intensity)
            
            # "Adaptive thinning", decreases upper bound           
            self.compute_intensity_upper_bound(time_current)
            intensity_hazard = np.sum(self.intensity_ub)
            
        return time_current

    
    def sample_one_event_sep(self):
        # Time of next event for each event type
        time_of_happen = np.zeros((self.dim_process,), dtype=dtype)
        
        # Looping over each event type in the process
        for event_type in range(self.dim_process):
            # Sample one event with event type event_type using "thinning algorithm"
            time_of_happen[event_type] = np.copy(self.sample_time_given_type(event_type))
            
        # Choose event that happens first
        time_since_start_new = np.min(time_of_happen)
        event_type_new = np.argmin(time_of_happen)
        
        return time_since_start_new, event_type_new


    def sample_one_event_tog(self):
        # Finds when next event happens using total intensity across event types
        time_since_start_new = self.sample_time_for_all_type()
        
        # Compute intensity given new time
        self.compute_intensity_given_past(
            time_since_start_new
        )
        
        # Transforms intensity of each event type to a probability
        prob = self.intensity / np.sum(self.intensity)
        
        #  Choose event type based off those probabilities 
        event_type_new = np.random.choice(range(self.dim_process), p = prob)
        
        return time_since_start_new, np.int32(event_type_new)

    
    def sample_one_event(self):
        # Samples events using total intensity or event intensity
        if self.sum_for_time:
            return self.sample_one_event_tog()
        else:
            return self.sample_one_event_sep()

  
    def gen_one_sequence(self, max_len):
        """
        Generates a single sequence of a Neural Hawkes process with inhibition.

        Parameters:
        - max_len (int): Pre-sampled value to set the maximum length of the sequence.

        This method generates a single sequence of a Neural Hawkes process using a thinning algorithm.
        It iteratively samples events, updates the sequence, and records relevant information about each event.
        The maximum length of the sequence is controlled by the pre-sampled value 'max_len'.
        """

        self.restart_sequence()
        
        # Initialize time overall and for each event type
        time_since_start = np.float32(0.0)
        time_since_start_each_event = np.zeros(
            (self.dim_process,), dtype=dtype
        )       
        
        # Looping over each event in the sequence:
        for event_idx in range(max_len):
            self.compute_hidden_states()
            
            time_since_start_new, event_type_new = self.sample_one_event()
            self.cnt_total_event += 1
            
            # Update sequence
            time_since_last_event = time_since_start_new - time_since_start
            time_since_start = time_since_start_new
            time_since_last_same_event = time_since_start - time_since_start_each_event[event_type_new]
            time_since_start_each_event[event_type_new] = time_since_start
            
            self.one_sequence.append({'event_idx': self.cnt_total_event,
                                         'event_type': event_type_new,
                                         'time_since_start': time_since_start,
                                         'time_since_last_event': time_since_last_event,
                                         'time_since_last_same_event': time_since_last_same_event})            
        
        # throw away the BOS item at the head of the sequence
        self.one_sequence.pop(0)

        
    def gen_sequences(self, sequence_settings):
        
        """
        Generate multiple sequences of the Hawkes process.

        Parameters:
        - sequence_settings (dict): A dictionary containing generation settings.
            - sequences: Number of sequences to generate.
            - min_len: Minimum length of each sequence.
            - max_len: Maximum length of each sequence.

        The method generates 'num_sequences' sequences of the Hawkes process, each with a random length
        between 'min_len' and 'max_len'.
        """
     
        num_sequences = sequence_settings['num_sequences']
        
        # List of generated sequences
        self.list_sequences = []
        # Current sequence to be generated
        sequence_cnt = 0
        
        while sequence_cnt < num_sequences:
            # Randomly select length of sequence 
            max_len = np.int32(
                round(np.random.uniform(low=sequence_settings['min_len'], high=sequence_settings['max_len']))
            )
            
            # Generate one sequence
            self.gen_one_sequence(max_len)
            # Save sequence
            self.list_sequences.append(self.one_sequence)
            sequence_cnt += 1
            
            # Print progress to user
            if sequence_cnt % 10 == 0:
                print( "idx sequence of gen : ", (sequence_cnt, self.name) )
                print( "total number of sequences : ", num_sequences )

                
    def save_sequences(self, file_save):
        with open(file_save, 'wb') as f:
            pickle.dump(self.list_sequences, f)
s