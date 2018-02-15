#filter sentences with users repeating in the same context

from collections import defaultdict
import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

adoption_sentence_filename = "/mnt/filer01/word2vec/degree_distribution/sentences_files/userSentencesComb_12hr"
out_file = adoption_sentence_filename+"_filter"
num_files = 13
NUM_PROCESSES = num_files

linecount=0
with open(adoption_sentence_filename, 'r') as fr, open(out_file, 'w') as fd:
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		s=set(u)
		if len(s)>1:
			fd.write(line+"\n")
		linecount+=1
		if linecount%1000000==0:
			print "path count", linecount
print "Sequence file read"

"""
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
	presentc = pickle.load(open(out_dir+"/frequencyNodes_1hr"+str(file_num)+".pickle","rb"))
	presentcc = pickle.load(open(out_dir+"/frequencyContextLength_1hr"+str(file_num)+".pickle","rb"))
	for i in presentc:
		count[i]+=presentc[i]
	for i in presentcc:
		context_count[i]+=presentcc[i]
pickle.dump(count,open(out_dir+"/comb_frequencyNodes_1hr_timeonly.pickle","wb"))
pickle.dump(context_count,open(out_dir+"/comb_frequencyContextLength_1hr_timeonly.pickle","wb"))	
"""
