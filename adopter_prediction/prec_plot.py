#plot MAP, precision, recall at k for different k in single tag case and averaged over 100 tags
#plot precision at k and R for individual topics with learned weights

from collections import defaultdict
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
"""
top_k = [25,50]+range(100,1001,100) #[100,200,300]+range(500,5001,500)

def eval_plot(vec,nbapp,fol,ylab,title):
	plt.plot(top_k, vec)
	plt.plot(top_k, nbapp)
	plt.plot(top_k, fol)
	plt.legend(['User vectors','Frequency','Followers'])
	plt.xlabel('k')
	# plt.xlim(xmin=top_k[0])
	plt.ylabel(ylab)
	plt.title(title)
	plt.grid()
	plt.show()
	
#single tag plot
with open("prec_plot_single1.pickle","rb") as fr:
	mapk = pickle.load(fr)
	preck = pickle.load(fr)
	reck = pickle.load(fr)
	vec,nbapp,fol = zip(*preck)
	eval_plot(list(vec),list(nbapp),list(fol),'Precision at k','Precision@k values at different k')

def average(eval_list):
	avg_vec = [0.0]*len(top_k)
	avg_nbapp = [0.0]*len(top_k)
	avg_fol = [0.0]*len(top_k)
	for l in eval_list:
		vec,nbapp,fol = zip(*l)
		for i,v in enumerate(vec):
			avg_vec[i]+=v
		for i,v in enumerate(nbapp):
			avg_nbapp[i]+=v
		for i,v in enumerate(fol):
			avg_fol[i]+=v
	num_tags = len(eval_list)
	avg_vec = [v*1.0/num_tags for v in avg_vec]
	avg_nbapp = [v*1.0/num_tags for v in avg_nbapp]
	avg_fol = [v*1.0/num_tags for v in avg_fol]
	return avg_vec,avg_nbapp,avg_fol
	
top_k = [1,2,5]+range(10,101,10)
#100 tags
with open("prec_plot_k.pickle","rb") as fr:
	mapk = pickle.load(fr)
	preck = pickle.load(fr)
	reck = pickle.load(fr)
	vec,nbapp,fol = average(mapk)
	eval_plot(vec,nbapp,fol,'Precision at k','Precision@k values at different k')
"""
num_bin = 50
def eval_plot(eval,rec,xlab,title):
	plt.hist(eval, num_bin)
	# plt.bar(range(1,len(eval)+1), eval)
	# plt.bar(range(1,len(rec)+1), rec)
	plt.xlabel(xlab)
	plt.ylabel('Frequency')
	plt.title(title)
	plt.grid()
	plt.show()

# with open("mean_precision_n10_rf_prec10.pickle","rb") as fr:
	# prec_k_total = pickle.load(fr)
	# cand_set_recall = pickle.load(fr)
	# cand_set_size_list = pickle.load(fr)

with open("eval_n10_lr.pickle","rb") as fr:
	ap_total = pickle.load(fr)
	prec_k_total = pickle.load(fr)
	_ = pickle.load(fr)
	cand_set_recall = pickle.load(fr)	

# print sum(cand_set_size_list)
# eval_plot(list(prec_k_total),list(cand_set_size_list),'Precision@k','Histogram of Prec@10 for 100 topics')
user,_,_ = zip(*prec_k_total)
eval_plot(list(user),[],'Precision@k','Histogram of Precision@500')