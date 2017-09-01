import numpy as np
import tensorflow as tf
import time
import pdb 
import os 
import socket
import multiprocessing
from multiprocessing import Process
from multiprocessing.pool import Pool
from rnns import *

def model_driver(d,data):
    """This function builds the computation graph for the specified model
    The input is the dictionary d with fields:
        model (model to be learnt), wform (form of the W matrices - full/diagonal/scalar/constant) 
        K (number of states), L1 (input dimensions), L2 (output dimensions), numlayers (number of layers)
    """
    # Reset graph
    tf.reset_default_graph()
    
    rnn1 = rnn( model_specs = d, initializer = d['init']) 
    rnn1_handles = rnn1.build_graph()  

    config = tf.ConfigProto(log_device_placement = False)
    config.gpu_options.allow_growth=True
    config.allow_soft_placement=True

    with tf.Session(config = config) as sess:
        sess.run(tf.initialize_all_variables())

        all_times, tr_logls, test_logls, valid_logls = rnn1.optimizer(data = data, rnn_handles = rnn1_handles, sess = sess) 

        
    max_valid = np.max(np.array(valid_logls)) #kind of unnecessary
    max_test = np.max( np.array(test_logls))
    res_dictionary = {'valid':  np.array(valid_logls), 
                  'max_valid':max_valid , 
                  'tst':  np.array(test_logls), 
                  'max_test':max_test, 
                  'tr': np.array(tr_logls), 
                  'all_times':all_times, 
                  'tnparams':rnn1.tnparams}

    res_dictionary.update(d)

    return res_dictionary

def main(dictionary):

    # first get the data and the resulting model parameters  
    data, parameters = load_data(dictionary = dictionary)
    dictionary.update(parameters) # add the necessary model parameters to the dictionary here
    
    ### next thing is determining the hyperparameters
    np.random.seed( dictionary['seedin'][0] )
    tf.set_random_seed( dictionary['seedin'][1] ) 
    timestamp = str(round(time.time()))

    # lower and upper limits for hyper-parameters to be sampled
    lr_min ,lr_max = dictionary['lr_min'], dictionary['lr_max']
    num_layers_min, num_layers_max = dictionary['num_layers_min'], dictionary['num_layers_max']
    K_min, K_max, min_params, max_params = return_Klimits(
            model = dictionary['model'], 
            wform = dictionary['wform'], 
            data = dictionary['data']) 

    records = [] #information will accumulate in this
    for i in range(dictionary['num_configs']):
        while True:  
            try:
                lr, K, num_layers, momentum = generate_random_hyperparams(
                        lr_min =  lr_min, lr_max = lr_max, 
                        K_min = K_min, K_max = K_max , 
                        num_layers_min = num_layers_min, 
                        num_layers_max = num_layers_max,
                        load_hparams = (dictionary['load_hparams'],i))
                
                print("Configuration ",i,
                      "K = ",K, 
                      "num_layers = ", num_layers,
                      "Learning Rate = ", lr,
                      "Momentum = ", momentum)  
                #this if clause enables the user to restart an experiment from a specific point the experiment
                if i < dictionary['start']:
                    break
                
                try: # Sometimes resources may get exhausted, this exception handles that 
                    dictionary.update({'LR': lr, 
                                        'K': K, 
                                        'num_layers': num_layers,
                                        'min_params': min_params,
                                        'max_params': max_params,
                                        'momentum' : momentum} ) 
                    run_info = model_driver(d = dictionary, data = data) 
                       
                    #append the performance records
                    records.append(run_info)
                except KeyboardInterrupt:
                    raise 
                except:
                    print('Resouces exhausted for this configuration, moving on')
                    #raise
                break
        
            except num_paramsError:
                print('This parameter configuration is not valid!!!!!!!') 

        # Save in directory
        savedir = 'experiment_data/'
        np.save( savedir + dictionary['server'] 
                + '_data_' + dictionary['data'] 
                + '_model_' + dictionary['model'] 
                + '_'+ dictionary['wform_global'] 
                + '_optimizer_' + dictionary['optimizer']
                +'_device_'+ dictionary['device']
                + '_' + timestamp, records)
    
    return records


#import matplotlib.pyplot as plt
wform = 'conv'# either diagonal or full
input_dictionary = {'seedin' : [1144, 1521], #setting the random seed. First is for numpy, second is for tensorflow 
            'task' : 'text', #this helps us how to load the data with the load_data function in rnns.py 
            'data' : 'ptb', #the dataset, options are inside the load_data function 
            'model': 'vector_w_conv', #options are: mod_lstm (our custom cell), lstm (tensor flow's lstm cell), gated_wf (our custom gru cell), gru (tensor flow's gru cell), mod_rnn (our custom rnn cell) 
            'wform' : wform, 
            'wform_global' : wform,
            'num_configs' : 60, #number of hyper parameter configurations to be tried 
            'start' : 0,  #this is used to start from a certain point (can be useful with fixed seed, or when hyper-parameters are loaded) 
            'EP' : 300, #number of epochs per run 
            'dropout' : [0.9, 0.9], #first is the input second is the output keep probability 
            'device' : 'gpu:0', #the device to be used in the computations 
            'server': socket.gethostname(),
            'verbose': False, #this prints out the batch location
            'load_hparams': False, #this loads hyper-parameters from a results file
            'count_mode': False, #if this is True, the code will stop after printing the number of trainable parameters
            'init':'xavier', #initialization method, options are 'xavier','random_unform' 
            'lr_min':-4, 'lr_max':-2, #the lower and upper limits for the exponent of the learning rate
            'num_layers_min':1, 'num_layers_max':3, #lower and upper limits for number of layers
            'optimizer':'RMSProp', #options are, Adam, RMSProp, Adadelta
            'notes':'I am trying the gated_wf model; with n layers in {1,3} on ptb. K is in range {200,...,400}, I have just added iterations over batches'}

perfs = main(input_dictionary)
