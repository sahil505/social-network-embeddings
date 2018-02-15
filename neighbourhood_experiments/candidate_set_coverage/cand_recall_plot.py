#plot candidate set recall for different n,c averaged over 100 or 50 tags

from collections import defaultdict
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

num_init_adopters = [10,100] #range(10,101,10)
top_k = np.arange(0.6,1.05,0.1) #np.arange(0.5,1.01,0.1) #[500,1000,2000]+range(3000,10001,1000)
seq_len_threshold = 500

def get_stats(u):
	return [np.mean(u), np.std(u)] #, np.median(u)

def cc_plot(cc):
	cc_mean, cc_std = zip(*cc)
	plt.errorbar(num_init_adopters, cc_mean, yerr=cc_std, fmt='o')
	plt.xlabel('Number of initial adopters, n')
	plt.ylabel('Proportion of adopters in candidate set')
	plt.title('Candidate set coverage with varying value of n\nc=1000, Avg. over 100 topics')
	plt.grid()
	plt.show()

def cc_boxplot(data,n,xlab,avg):
	plt.boxplot(data, labels=xlab, whis='range', showmeans=True, meanprops=dict(marker='D', markerfacecolor='red'))
	plt.xlabel('Number of nearest neighbours queried, c\n(with average neighbour set size)')
	# plt.ylabel('Neighbourhood coverage')
	# plt.title('Candidate set coverage with varying value of c\nn='+str(n)+', Average over 50 topics')
	
	# plt.xlabel('Radius for querying neighbour set, r\n(with average neighbour set size)')
	# plt.ylabel('Neighbourhood coverage')
	# plt.title('Candidate set coverage with varying value of r\nn='+str(n)+', Average over 50 topics')
	
	# plt.ylabel('Fraction of neighbouring users from same geography')
	# plt.title('Geography precision with radius based search\nAverage over '+str(n)+' users')

	plt.ylabel('Fraction of friends in neighbour set')
	plt.title('Following Coverage with nearest neighbour search\nAverage number of friends '+str(round(avg,2))+', Average over '+str(n)+' users')
	
	plt.tight_layout()
	plt.ylim(0,1.0)
	plt.grid()
	plt.show()

def cc_scatterplot(cc,spread):
	plt.scatter(cc, spread)
	m, b = np.polyfit(cc,spread, 1)
	plt.plot(np.asarray(cc), m*np.asarray(cc) + b, 'r-', label = 'Linear fit')
	plt.ylim(-0.1,1.0)
	plt.xlim(-0.1,1.0)
	# plt.yscale('log')
	# plt.xscale('log')
	# plt.ylim(500,plt.ylim()[1])
	# plt.xlabel('Proportion of first 1000 adopters present in candidate set')
	# plt.ylabel('Total spread')
	# plt.title('Candidate set coverage and eventual spread of topics\nCorr. coeff. = '+str(round(np.corrcoef(cc,spread)[0,1],4)))	
	# plt.xlabel('Proportion of adopters in candidate set')
	# plt.ylabel('Precision@10')
	# plt.title('Candidate set coverage and Precision@10 of topics\nCorr. coeff. = '+str(round(np.corrcoef(cc,spread)[0,1],4)))
	plt.xlabel('Network neighbours (followers)')
	plt.ylabel('Vector space neighbours')
	# plt.title('Likelihood of co-adoption of users with different neighbourhoods\nCorr. coeff. = '+str(round(np.corrcoef(cc,spread)[0,1],4)))
	plt.grid()
	plt.legend(loc='upper left')
	plt.show()

num_bin = 10
def ent_histogram(val):
	x,bins,_=plt.hist(val, num_bin, rwidth=0.8, align='left')
	# plt.bar(range(1,len(val)+1), val)
	# plt.bar(range(1,len(rec)+1), rec)
	plt.xlim(-0.1,1.0)
	plt.xlabel('Precision@10')
	plt.ylabel('Frequency')
	# plt.title('Entropy of distribution of geo-locations in clusters')
	plt.tight_layout()
	plt.grid()
	plt.show()

def freq_plot(y):
	time_bins = [(0,5),(6,8),(9,11),(12,14),(15,17),(18,20),(21,23)]
	x = [str(i)+'-'+str(j) for i,j in time_bins]
	plt.bar(range(len(y)), y, align='center')
	plt.xticks(range(len(y)), x, size='small')
	plt.xlabel('Time of day (in hour)')
	# plt.xlim(xmin=0)
	plt.ylabel('Proportion of tweets')
	plt.title('Frequency distribution of tweeting time')
	plt.grid()
	plt.show()
#coverage box plots
"""
for n in num_init_adopters:
	# cc = []
	cand_size = []
	data = []
	for i in top_k:
		# with open("candset_stat_files/candset_n"+str(n)+"_c"+str(i)+".pickle","rb") as fr:
		with open("candset_stat_files/candset_n"+str(n)+"_r"+str(i)+".pickle","rb") as fr:
			cand_set_recall = pickle.load(fr)
			cand_set_overlap = pickle.load(fr)
			cand_set_cr = pickle.load(fr)
			cand_set_size_list = pickle.load(fr)
		print n,i,len(cand_set_recall)
		# if i==1000:
		if n==10:
			cand_set_recall = cand_set_recall[:50]
		# cc.append(get_stats(cand_set_recall))	
		data.append(cand_set_recall)
		cand_size.append(np.mean(cand_set_size_list))
	xlab = [str(x)+'\n('+str(y)+')' for x,y in zip(top_k,cand_size)]
	cc_boxplot(data,n,xlab)

# cc_plot(cc)
# cc_boxplot(data)
"""
#entropy histogram plots
"""
with open("user_vector_cluster_entropy.pickle","rb") as fr:
	_ = pickle.load(fr)
	c_ent = pickle.load(fr)
	_ = pickle.load(fr)
	l_ent = pickle.load(fr)
ent_histogram(c_ent)
ent_histogram(l_ent)
"""

#coverage vs spread scatter plots
"""
# with open("candset_stat_files/test_sequence_indices_thr1000.pickle","rb") as fr:
# 	seq_index_filter = pickle.load(fr)
NUM_LINES = 2000#len(seq_index_filter)

num_workers = 9
lines_per_process = int(NUM_LINES/(2*num_workers))
cand_set_recall_spread = []
for s,e in ( (i,min(i+lines_per_process,NUM_LINES)) for i in xrange(0,NUM_LINES,lines_per_process) ):
	print s,e
	with open("candset_stat_files/candset_vs_spread_n"+str(num_init_adopters)+"_c"+str(top_k)+"_seq"+str(seq_len_threshold)+"_ex"+str(s)+".pickle","rb") as fr:
		cc_subset = pickle.load(fr)
		cand_set_recall_spread += cc_subset
# cand_set_recall_spread = sorted(cand_set_recall_spread, key=lambda x: x[0])
cc,spread = zip(*cand_set_recall_spread)
print sum(cc)*1./len(cc), sum(spread)*1./len(spread), min(cc), max(cc), min(spread), max(spread)
cc_scatterplot(cc,spread)
"""

#sliding window median plots
"""
window_length = 50
median_spread_mw = []
median_cc_mw = []
K=2000
for i in range(0,len(cand_set_recall_spread)):
	m = np.median(spread[i:i+window_length])
	# m = 0
	# for s in spread[i:i+window_length]:
	# 	if s>=K:
	# 		m+=1
	# m = m*1./window_length
	mw = cc[i:i+window_length]
	c = mw[len(mw)//2]
	median_spread_mw.append(m)
	median_cc_mw.append(c)
cc_scatterplot(median_cc_mw,median_spread_mw)
"""

#prec@10 vs coverage scatter plots

with open("candset_stat_files/nbr_frac0.5_seq100.pickle","rb") as fr:
	cand_set_recall = pickle.load(fr)
	cand_set_overlap = pickle.load(fr)
	cand_set_size_list = pickle.load(fr)
print np.mean(cand_set_recall,axis=0), len(cand_set_recall)
print cand_set_recall[0:10], cand_set_overlap[0:10], cand_set_size_list[0:10]
fol,vec = zip(*cand_set_recall)
# nbh,_ = zip(*cand_set_size_list)
# p_fol = [x*1./y for (x,y) in zip(fol,nbh)]
# p_vec = [x*1./y for (x,y) in zip(vec,nbh)]
# print np.mean(p_fol), np.mean(p_vec)
cc_scatterplot(fol,vec)
# ent_histogram(u)

"""
#entropy vs spread scatter plots
with open("candset_stat_files/entropy_vs_spread_seq500_ex0.pickle","rb") as fr:
	ent_spread = pickle.load(fr)
ent_spread = sorted(ent_spread, key=lambda x: x[1])
e,s,er = zip(*ent_spread)
cc_scatterplot(er,s)
"""
"""
#activity time histogram plots
with open("candset_stat_files/sample_user_activity_time_uneven_bins.pickle","rb") as fr:
	sample_activity = pickle.load(fr)
	# total_tweets = pickle.load(fr)
# total_activity_bins = [i*1./total_tweets for i in total_activity_bins]
# print total_tweets, total_activity_bins
print len(sample_activity)
for i in sample_activity:
	c = sample_activity[i]
	freq = [0]*7
	for b in c:
		freq[b]+=c[b]
	freq_plot(freq)
"""

#geography, follower, following coverage box plots
"""
cand_size = []
avg=[]
data = []
top_k = [1000,2000,5000,10000] #np.arange(0.6,1.05,0.1).tolist()+[1.2] #np.arange(0.6,1.15,0.1)
n=10000
for i in top_k:
	# with open("candset_stat_files/candset_loc_c"+str(i)+".pickle","rb") as fr:
	with open("candset_stat_files/candset_fol_fr_c"+str(i)+".pickle","rb") as fr:
		cand_set_recall = pickle.load(fr)
		cand_set_overlap = pickle.load(fr)
		num_nbr = pickle.load(fr)
		cand_set_size_list = pickle.load(fr)
	print i,len(cand_set_recall)
	# cc_geo,cc_prec_geo = zip(*cand_set_recall)
	cc_fol,cc_fr = zip(*cand_set_recall)
	num_fol,num_fr = zip(*num_nbr)
	avg_fol = np.mean(num_fol)
	avg_fr = np.mean(num_fr)
	avg.append(avg_fr)
	print avg_fol,avg_fr
	# cc.append(get_stats(cand_set_recall))	
	data.append(cc_fr)
	cand_size.append(np.mean(cand_set_size_list))
print np.mean(avg)
xlab = [str(x)+'\n('+str(y)+')' for x,y in zip(top_k,cand_size)]
cc_boxplot(data,n,xlab,np.mean(avg))
"""