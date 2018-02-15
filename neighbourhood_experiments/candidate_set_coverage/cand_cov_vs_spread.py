#get candidate set coverage in first 1000 adopters for topics with atleast 1000 adopters and compare with eventual spread

import cPickle as pickle
import time
from math import sqrt
import random
from heapq import nsmallest, nlargest, merge
import numpy
# from scipy.spatial import cKDTree as KDTree
from sklearn.neighbors import NearestNeighbors
import sys
from multiprocessing import Pool, cpu_count

NUM_PROCESSES = 9

vec_file = "/mnt/filer01/word2vec/node_vectors_1hr_pr.txt"
adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt" #"sample_sequences"
num_init_adopters = 100
top_k = 4000
seq_len_threshold = 500 #top_k
cand_size_factor = 1
train_ex_limit = 100
norm_vec = True

print vec_file, num_init_adopters, top_k, seq_len_threshold, norm_vec

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

# building kd-tree
tic = time.clock()
# kd = KDTree(vec, leafsize=10)
neigh = NearestNeighbors(n_neighbors=5, radius=1.0, algorithm='ball_tree', leaf_size=100, metric='minkowski', p=2) #'ball_tree', 'kd_tree', 'auto'
neigh.fit(vec)
toc = time.clock()
print "ball tree built in", (toc-tic)*1000

def get_candidate_set(query_set,next_adopters,N):
	try:
		query_set_ind = [ vocab_index[query] for query in query_set ]
	except KeyError:
		print "query word not present"
		return
	query_vec = [vec[i] for i in query_set_ind]
	# query using scipy kdtree
	# d_list,knn_list = kd.query(query_vec,k=cand_size_factor*N+len(query_set_ind))
	# query using sklearn
	d_list,knn_list = neigh.kneighbors(X=query_vec, n_neighbors=cand_size_factor*N+len(query_set_ind), return_distance=True)

	cand_set = set()
	for index_list in knn_list:
		filtered=[idx for idx in index_list if idx not in query_set_ind]
		cand_set.update(filtered)

	cand_set_size = len(cand_set)
	M = len(next_adopters)
	next_adopters_index = [vocab_index[a] for a in next_adopters]
	next_adopters_index = set(next_adopters_index)
	num_adopters = len(cand_set&next_adopters_index)

	# print "candidate set recall", num_adopters, "out of", len(next_adopters), "cand size", len(cand_user_idx)
	cr = num_adopters*1./cand_set_size
	cc = 0.0
	if M!=0:
		cc = num_adopters*1./M
	return num_adopters, cand_set_size, cc, cr, M

def print_stats(u):
	return [numpy.mean(u), numpy.std(u), numpy.median(u)]

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
seq_index_filter = seq_index_filter[0:2000]

def cand_set_stat(process_num,start,end,num_init,num_query):
	print process_num, start, end, num_init, num_query
	cand_set_recall_spread = []
	cand_set_overlap = []
	cand_set_cr = []
	cand_set_size_list = []
	cand_cov = 0.0
	cand_cr = 0.0
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
		init_adopters=seq_sample_vocab[0:num_init]
		next_adopters = seq_sample_vocab[num_init:seq_len_threshold]
		N = num_query #1000 #M #num_users

		op, cand_set_size, cc, cr, M = get_candidate_set(init_adopters, next_adopters, N)
		cand_cov+=cc
		cand_cr+=cr

		cand_set_recall_spread.append((cc,total_spread))
		cand_set_overlap.append(op)
		cand_set_cr.append(cr)
		cand_set_size_list.append(cand_set_size)

		# print "cc", cc, "cand size", cand_set_size, "Avg", cand_cov*1./(l+1), sum(cand_set_size_list)*1./(l+1), "adop in cand", op, "total", M, total_spread, l
		l+=1
		# if l==train_ex_limit:
		# 	break
	print process_num, start, num_init, num_query, "num examples", l, "cc", print_stats(cand_set_recall_spread), avg_num_adopters*1./l
	with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/candset_vs_spread_n"+str(num_init)+"_c"+str(num_query)+"_seq"+str(seq_len_threshold)+"_ex"+str(start)+".pickle","wb") as fd:
		pickle.dump(cand_set_recall_spread,fd)
		# pickle.dump(cand_set_overlap,fd)
		# pickle.dump(cand_set_cr,fd)
		# pickle.dump(cand_set_size_list,fd)

# tic = time.clock()
# cand_set_stat(0,num_init_adopters,top_k)
# toc = time.clock()
# print "cand set eval in", (toc-tic)*1000

NUM_LINES = len(seq_index_filter)

num_workers = min(NUM_PROCESSES,cpu_count())
pool = Pool(processes=num_workers) 
process_num=0
lines_per_process = int(NUM_LINES/(2*num_workers))
for s,e in ( (i,min(i+lines_per_process,NUM_LINES)) for i in xrange(0,NUM_LINES,lines_per_process) ):
	pool.apply_async(cand_set_stat, args=(process_num,s,e,num_init_adopters,top_k))
	process_num+=1
pool.close()
pool.join()

print vec_file, num_init_adopters, top_k