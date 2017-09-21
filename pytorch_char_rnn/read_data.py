import pickle
import os 
import operator 
import numpy as np
from matplotlib import pyplot as plt

vanilla_diag = []
vanilla_full = []
for fn in os.listdir('vanilla_diag'):
	if os.path.isfile('vanilla_diag/' + fn):
		vanilla_diag.append(pickle.load(open('vanilla_diag/' + fn,'rb')))

for fn in os.listdir('vanilla_full'):
	if os.path.isfile('vanilla_full/' + fn):
		vanilla_full.append(pickle.load(open('vanilla_full/' + fn,'rb')))

def violin_plot(inpt, T, npts, N, topN):
	colors = 'mb'
	line_color = 'g'
	ss_ids = np.linspace(0,T-1,num=npts,dtype = 'int64')
	for i, model in enumerate(inpt):
		sorted_data = sorted(model, key=lambda k: k['final_test_loss']) 
		picks = sorted_data[0:topN]
		all_test_loss = []
		for run in picks:
			all_test_loss.append(run['test_loss'])
		all_test_loss = np.stack(all_test_loss)[:,ss_ids]
		mean_test_loss = all_test_loss.mean(0)
		violin_parts = plt.violinplot(all_test_loss, positions=ss_ids, widths=5, showmeans=False, showmedians=True, showextrema=False)
		for body in violin_parts['bodies']:
			body.set_color(colors[i])
		violin_parts['cmedians'].set_color(line_color)
		violin_parts['cmedians'].set_linewidth(3)
		plt.plot(ss_ids, mean_test_loss, '-'+colors[i], label=model[0]['model']) 
		plt.legend(loc=1)

fig = plt.figure(num=None, figsize=(9, 5), dpi=100, facecolor='w', edgecolor='k')
violin_plot([vanilla_diag, vanilla_full], 100, 15, 60, 6)
plt.show()