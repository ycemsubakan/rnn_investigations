import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pdb

def getperformancelist(data, model, wform, directory):
    in_directory = 'experiment_data/' +  directory
    filekey = 'data_'+data+'_model_'+model+'_'+wform+'_'
    file_list = [filename for filename in os.listdir(in_directory) if filekey in filename]    
    
    mergeddata = []
    for file_name in file_list:
        mergeddata.extend( list( np.load( in_directory + file_name)))
    return mergeddata

def makeviolinplot(*arg):

    dataset = arg[-1]['dataset']
    if dataset == 'mnist':
        xlabel = 'Training iterations (in sec)'
        ylabel = 'Test Accuracy'
        yrange = [0.9,1 ]
    elif dataset in ['JSBChorales','Piano-midi','Nottingham','MuseData']:
        xlabel = 'Training iterations'
        ylabel = '-Log Likelihood'
        if dataset == 'JSBChorales':
            yrange = [8,12]
        elif dataset == 'Piano-midi':
            yrange = [7,12]
        elif dataset == 'Nottingham':
            yrange = [3,8]
        elif dataset == 'MuseData':
            yrange = [7,12]
    
    T, npts, N, topN  = arg[-1]['T'], arg[-1]['npts'], arg[-1]['N'], arg[-1]['topN']
    ss_ids = np.linspace(0,T-1,num=npts,dtype = 'int64')
        
    model_infos = arg[:-1]
    
    picks = []
    
    for info in model_infos:
        all_y = [info[j]['valid'].reshape(1,T) for j in range(N)]  
        all_y_mat = np.concatenate(all_y,axis=0)[:,ss_ids]

        if dataset in ['JSBChorales','Piano-midi','Nottingham','MuseData']:
            picks.append( list(np.argsort(all_y_mat[:,-1])[0:topN]) ) 
        else:
            picks.append( list(np.flip(np.argsort(all_y_mat[:,-1]),0)[0:topN]) ) 

    colors = 'mb'
    line_color = 'g'
    for i, info in enumerate(zip(model_infos,picks)):
        all_y = [info[0][j]['tst'].reshape(1,T) for j in info[1]]  
        min_logl = min([all_y_row[0,-1] for all_y_row in all_y])
        print('The achieved minimum N.Likelihood for ', info[0][0]['wform'],' model is ', min_logl) 

        all_y_mat = np.concatenate(all_y,axis=0)[:,ss_ids]
        mean_y = all_y_mat.mean(0)  
        #all_x = [np.cumsum(info[0][j]['all_times']).reshape(1,T) for j in info[1]]
        #all_x_mat = np.concatenate(all_x,axis=0)[:,ss_ids]

        n_params = np.mean([info[0][inds]['tnparams'] for inds in info[1]])
       
        violin_parts = plt.violinplot(all_y_mat, positions = ss_ids, widths = 5, showmeans = False, showmedians = True, showextrema=False)
        for body in violin_parts['bodies']:
            body.set_color(colors[i])
        violin_parts['cmedians'].set_color(line_color)
        violin_parts['cmedians'].set_linewidth(3)

        #violin_parts['cmins'].set_color(line_color)
        #violin_parts['cmaxes'].set_color(line_color)




        plt.plot( ss_ids, mean_y, '-' + colors[i] , label =  info[0][0]['wform'] + ', #p = ' + str(round(n_params,0))) 
        plt.ylim(yrange)
        plt.xlim( [0,T] ) 
        
        #title = 'K = ' + str(info[0][n]['K']) + ', num layers = ' + str(info[0][n]['num_layers']) + ', LR = ' +  str(round(info[0][n]['LR'],5) ) 
        if 'lstm' in info[0][0]['model']:
            plt.title( 'LSTM' )
        elif 'gated_wf' in info[0][0]['model']:
            plt.title( 'GRU' )
        elif 'rnn' in info[0][0]['model']:
            plt.title( 'Vanilla RNN' )


        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        #plt.xticks(np.arange(1,T+1),np.arange(0,T))

        plt.legend(loc= 1)
        #plt.show()

# model options: mod_lstm, gated_wf, mod_rnn

matplotlib.rcParams.update({'font.size': 20})

model = 'gated_wf'
dataset = 'MuseData'
load_dataset = 'MuseData'

directory = 'temp_data/'

model1 = getperformancelist(data = load_dataset, model = model, wform = 'full', directory = directory)
model2 = getperformancelist(data = load_dataset, model = model, wform = 'diagonal', directory = directory)
print("Number of completed configurations for model 1 = ",len(model1),
      "Number of completed configurations for model 2 = ",len(model2))

T = 300 # max number of iterations
npts = 15# number iteration points to be subsampled
N = 58#number of configurations to be used in the plot
topN = 6 #number of top configurations 

fig = plt.figure(num=None, figsize=(9, 5), dpi=100, facecolor='w', edgecolor='k')
makeviolinplot(model1, model2, {'dataset':dataset, 'T':T, 'npts':npts, 'N':N, 'topN':topN})

#fig.savefig('../WASPAA2017/paper/'+ model + dataset  +  '_RMSprop.eps', dpi=100)

plt.show()



