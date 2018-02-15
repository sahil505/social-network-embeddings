#count frequency of nodes occurring in sentences

from collections import defaultdict
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
"""
in_dir = "/mnt/filer01/word2vec/degree_distribution/sentences_files_timeonly/"
out_dir = "/mnt/filer01/word2vec/degree_distribution/count_files/"
num_files = 8
NUM_PROCESSES = num_files

def count_sentence_file(file_num,process_num):
	count=defaultdict(int)
	context_count=defaultdict(int)
	with open(in_dir+'/hashtagAdoptionSentences'+str(file_num)+'.txt','rb') as fr:
		for line in fr:
			line = line.rstrip()
			u = line.split(' ')
			s = len(u)
			context_count[s]+=1
			for id in range(0, s):
				author = int(u[id])
				count[author]+=1
	print "process", process_num, "file complete", file_num
	pickle.dump(count,open(out_dir+"/frequencyNodes_1hr_bfsr_timeonly"+str(file_num)+".pickle","wb"))
	pickle.dump(context_count,open(out_dir+"/frequencyContextLength_1hr_bfsr_timeonly"+str(file_num)+".pickle","wb"))

#run count_sentence_file on different adoption sentences files in parallel processes
num_workers = min(NUM_PROCESSES,cpu_count())
pool = Pool(processes=num_workers) 
process_num=0
for i in range(0,num_files):
	pool.apply_async(count_sentence_file, args=(i,process_num))
	process_num+=1
pool.close()
pool.join()	

#combine counts from different pickle file
count=defaultdict(int)
context_count=defaultdict(int)
for file_num in range(0,num_files):
	presentc = pickle.load(open(out_dir+"/frequencyNodes_1hr_bfsr_timeonly"+str(file_num)+".pickle","rb"))
	presentcc = pickle.load(open(out_dir+"/frequencyContextLength_1hr_bfsr_timeonly"+str(file_num)+".pickle","rb"))
	for i in presentc:
		count[i]+=presentc[i]
	for i in presentcc:
		context_count[i]+=presentcc[i]
pickle.dump(count,open(out_dir+"/comb_frequencyNodes_1hr_bfsr_timeonly.pickle","wb"))
pickle.dump(context_count,open(out_dir+"/comb_frequencyContextLength_1hr_bfsr_timeonly.pickle","wb"))	
"""
#plot
count = pickle.load(open("sentences_frequency_files/comb_frequencyNodes_1hr_bfsr_loc.pickle","rb"))
node_freq = []
for i in count:
	node_freq.append(count[i])
	
node_freq = np.array(node_freq)

num_bin = 100000
def freq_plot(data,xlab):
	values, base = np.histogram(data, bins=num_bin)
	cumulative = np.cumsum(values)
	# plt.plot(base[:-1], values, c='red') #frequency
	# plt.plot(base[:-1], cumulative/float(len(data)), c='red') #normalised
	# plt.plot(base[:-1], len(data)-cumulative, c='red') #inverse, greater than
	plt.plot(base[:-1], len(data)-np.append(0,cumulative)[:-1], c='red') #inverse, greater than or equal to
	plt.yscale('log')
	plt.xscale('log')
	plt.xlabel(xlab)
	plt.xlim(xmin=0)
	plt.ylabel('number of users')
	plt.title('cumulative frequency distribution (greater than or equal to)')
	plt.grid()
	plt.show()
	
freq_plot(node_freq,'Count of user occurrence in sentences')

#frequency plot of path lengths
ccount = pickle.load(open("sentences_frequency_files/comb_frequencyContextLength_1hr_bfsr_loc.pickle","rb"))
clength_freq = []
for i in ccount:
	clength_freq.append((i,ccount[i]))

def freq_plot_clength(data,xlab):
	x,y = zip(*data)
	x = [i-0.4 for i in x] #label at bar centre
	y = [i/float(sum(y)) for i in y] #normalised
	plt.bar(x, y) #frequency
	plt.xlabel(xlab)
	plt.xlim(xmin=0)
	plt.ylabel('Proportion of paths')
	plt.title('frequency distribution')
	plt.grid()
	plt.show()
freq_plot_clength(clength_freq,'Path length')

