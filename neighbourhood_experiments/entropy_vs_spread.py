#get entropy of distribution of first 1000 adopters in different clusters for topics with atleast 1000 adopters and compare with eventual spread

import cPickle as pickle
import time
from math import sqrt, log
import random
from heapq import nsmallest, nlargest, merge
import numpy
# from scipy.spatial import cKDTree as KDTree
from sklearn.neighbors import NearestNeighbors
import sys
from multiprocessing import Pool, cpu_count
from collections import defaultdict
import traceback

NUM_PROCESSES = 1

vec_file = "/mnt/filer01/word2vec/node_vectors_1hr_pr.txt"
adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt" #"sample_sequences"
seq_len_threshold = 500 #top_k
train_ex_limit = 100
norm_vec = True

print vec_file, seq_len_threshold, norm_vec

with open("/mnt/filer01/word2vec/degree_distribution/sequence_file_split_indices.pickle","rb") as fr:
	_ = pickle.load(fr)
	test_seq_id = pickle.load(fr)
test_seq_id = set(test_seq_id)

def read_vector_file(path_vectors_file):
	vocab = []
	vectors = []
	with open(path_vectors_file,"rb") as fr:
		_,dim = next(fr).rstrip().split(' ')
		word_vector_dim = int(dim)
		next(fr)
		for line in fr:
			line = line.rstrip()
			u = line.split(' ')
			if len(u) != word_vector_dim+1:
				print "vector length error"
			word = int(u[0])
			#normalise to length 1
			if norm_vec:
				vec = []
				length = 0.0
				for d in u[1:]:
					num=float(d)
					vec.append(num)
					length+=num**2
				#vec = map(float,u[1:])
				#length = sum(x**2 for x in vec)
				length = sqrt(length)
				vec_norm = [x/length for x in vec]
				vectors.append(vec_norm)
			else:
				vec = map(float,u[1:])
				vectors.append(vec)
			vocab.append(word)
	return vectors, vocab, word_vector_dim

vec,vocab,dim = read_vector_file(vec_file)
vocab_index=dict()
for i in xrange(0,len(vocab)):
	vocab_index[vocab[i]]=i
num_users = len(vocab)
print "num users in train sequences", num_users
# print "users removed from vocab", len(set(users_train)-set(vocab))
# print "users in test sequences but not in vocab", len(users_test-set(vocab))

def print_stats(u):
	e,t,r = zip(*u)
	return [numpy.mean(e), numpy.std(e), numpy.median(e)], [numpy.mean(t), numpy.std(t), numpy.median(t)], [numpy.mean(r), numpy.std(r), numpy.median(r)]

# reading test sequences
not_found_vocab=[]
# source_thr = 1395858601 + 12*60*60
# non_emergent_tags = pickle.load(open("/mnt/filer01/word2vec/degree_distribution/nonEmergentHashtags.pickle","rb"))
tag_seq = []
count=0
# nb_seq = dict()
# adlen = []
with open(adoption_sequence_filename, "rb") as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		not_found = set()
		adopters = set()
		# first_timestamp = int(u[1][0:u[1].index(',')])
		# first tweet only after source_thr timestamp
		# if first_timestamp>=source_thr
		# check if <5 tweets in 12 hours for emergent hashtags, not already popular
		# u[0] not in non_emergent_tags and
		if count in test_seq_id:
			seq=[]
			for i in xrange(1, len(u)):
				#timestamp = int(u[i][0:u[i].index(',')])
				author = int(u[i][u[i].index(',')+1 : ])
				if author in vocab_index:
					# removing repeat adopters
					if author not in adopters:
						seq.append(author)
						adopters.add(author)
				else:
					not_found.add(author)
			if len(seq)>0:
				tag_seq.append(seq)
				not_found_vocab.append(len(not_found))
				# adlen.append(len(seq))
		# elif count not in test_seq_id:
		# 	adop=[]
		# 	for i in xrange(1, len(u)):
		# 		author = int(u[i][u[i].index(',')+1 : ])
		# 		if author in vocab_index:
		# 			adop.append(author)
		# 	for author in set(adop):			
		# 		try:
		# 			nb_seq[author]+=1
		# 		except KeyError:
		# 			nb_seq[author]=1
		count+=1
#nb, number of training sequences participated in
# nb_seq_part = [(a,nb_seq[a]) for a in nb_seq]
# nb_seq_part_sorted = sorted(nb_seq_part, key=lambda x: x[1], reverse=True)
# nb_seq_order = [a for a,_ in nb_seq_part_sorted]
# pickle.dump(nb_seq_order,open(nb_sorted_pickle,"wb"))
# pickle.dump(adlen,open("adlen.pickle","wb"))

print len(tag_seq),len(test_seq_id),count
print sum(not_found_vocab)/float(len(not_found_vocab)),max(not_found_vocab),min(not_found_vocab)

"""
#test sequences in random order
seq_random_index=range(0,len(tag_seq))
random.shuffle(seq_random_index)

seq_index_filter = []
for i in seq_random_index:
	seq_sample_vocab = tag_seq[i]
	M = len(seq_sample_vocab)
	if M<seq_len_threshold:
		continue
	seq_index_filter.append(i)
print "tags remaining", len(seq_index_filter)

with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/test_sequence_indices_thr"+str(seq_len_threshold)+".pickle","wb") as fd:
	pickle.dump(seq_index_filter,fd)
"""
with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/test_sequence_indices_thr"+str(seq_len_threshold)+".pickle","rb") as fr:
	seq_index_filter = pickle.load(fr)
print "tags remaining", len(seq_index_filter)
# seq_index_filter = seq_index_filter[0:2000]

#get k-means clusters
with open("/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/user_vector_cluster_index_1hr_pr.pickle","rb") as fr:
	vec_cluster_idx = pickle.load(fr)

def get_entropy(adop):
	clusters = defaultdict(int)
	for u in adop:
		clusters[vec_cluster_idx[vocab_index[u]]]+=1
	ent=0.0
	for c in clusters:
		p = clusters[c]*1./seq_len_threshold
		if p > 0:
			ent+= -1.0*p*log(p,2)
	return ent

def init_adopt_stat(process_num,start,end):
	print process_num, start, end
	try:
		cand_set_recall_spread = []
		l=0
		avg_num_adopters = 0
		count=0
		for i in seq_index_filter:
			if count < start:
				count+=1
				continue
			elif count >= end:
				break
			count+=1
			seq_sample_vocab = tag_seq[i]
			total_spread = len(seq_sample_vocab)
			avg_num_adopters+=total_spread
			
			init_adopters=seq_sample_vocab[0:seq_len_threshold]
			ent = get_entropy(init_adopters)

			#entropy of random sample of users
			random_adopters = random.sample(vocab,seq_len_threshold)
			ent_M1 = get_entropy(random_adopters)

			ent_rel = 0.0
			if ent_M1>0:
				ent_rel = ent*1./ent_M1

			cand_set_recall_spread.append((ent,total_spread,ent_rel))

			l+=1
			if l%25==0:
				print "entropy", ent, "random", ent_M1, "total spread", total_spread, "rel", ent_rel, l
			# if l==train_ex_limit:
			# 	break
		print process_num, start, "num examples", l, "ent", print_stats(cand_set_recall_spread), avg_num_adopters*1./l
		with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/entropy_vs_spread_seq"+str(seq_len_threshold)+"_ex"+str(start)+".pickle","wb") as fd:
			pickle.dump(cand_set_recall_spread,fd)
	except Exception as e:
		print traceback.format_exc()

tic = time.clock()
init_adopt_stat(0,0,1000)
toc = time.clock()
print "init adopt eval in", (toc-tic)*1000

# NUM_LINES = len(seq_index_filter)

# num_workers = min(NUM_PROCESSES,cpu_count())
# pool = Pool(processes=num_workers) 
# process_num=0
# lines_per_process = int(NUM_LINES/(2*num_workers))
# for s,e in ( (i,min(i+lines_per_process,NUM_LINES)) for i in xrange(0,NUM_LINES,lines_per_process) ):
# 	pool.apply_async(init_adopt_stat, args=(process_num,s,e))
# 	process_num+=1
# pool.close()
# pool.join()

print vec_file