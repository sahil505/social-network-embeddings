#get nearest users to the initial adopters of a hashtag sequence in test sequences using user vectors and write candidate set size stats
#for different values of init adopters, query size or query radius
#changed index file of sequences with non-zero number of adopters (and those who are present in vocab) in sequence_file_split_indices.pickle

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
import traceback

NUM_PROCESSES = 2

vec_file = "/mnt/filer01/word2vec/node_vectors_1hr_pr.txt"
adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt" #"sample_sequences"
num_init_adopters = [10] #range(10,101,10)
top_k = [6000,10000] #range(4000,10001,2000)
query_rad = numpy.arange(0.6,1.05,0.05)
seq_len_threshold = 500 #top_k
cand_size_factor = 1
train_ex_limit = 50
norm_vec = True

print vec_file, num_init_adopters, top_k, train_ex_limit, query_rad, norm_vec

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
	
	# get vectors within distance N
	# _,knn_list = neigh.radius_neighbors(X=query_vec, radius=N, return_distance=True)
	qresult_size = []

	cand_set = set()
	for index_list in knn_list:
		qresult_size.append(len(index_list))
		filtered=[idx for idx in index_list if idx not in query_set_ind]
		cand_set.update(filtered)

	med_qresult_size = numpy.median(qresult_size)
	cand_set_size = len(cand_set)
	M = len(next_adopters)
	next_adopters_index = [vocab_index[a] for a in next_adopters]
	next_adopters_index = set(next_adopters_index)
	num_adopters = len(cand_set&next_adopters_index)
	cand_adopters = cand_set&next_adopters_index

	# print "candidate set recall", num_adopters, "out of", len(next_adopters), "cand size", len(cand_user_idx)
	cr = 0.0
	if cand_set_size!=0:
		cr = num_adopters*1./cand_set_size
	cc = 0.0
	if M!=0:
		cc = num_adopters*1./M
	return num_adopters, cand_set_size, cc, cr, M, med_qresult_size, cand_adopters, cand_set, next_adopters_index

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
	init_adopters=seq_sample_vocab[0:num_init_adopters]
	seq_sample_vocab = set(seq_sample_vocab[num_init_adopters:])
	M = len(seq_sample_vocab)
	N = top_k #1000 #M #num_users
	if M<seq_len_threshold:
		continue
	seq_index_filter.append(i)
print "tags remaining", len(seq_index_filter)

#train-test split for learning weights
num_train = int(0.5*len(seq_index_filter))
print "training examples present", num_train
train_seq_id_weight = seq_index_filter[:num_train]
test_seq_id_weight = seq_index_filter[num_train:]
with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/sequence_file_split_indices.pickle","wb") as fd:
	pickle.dump(train_seq_id_weight,fd)
	pickle.dump(test_seq_id_weight,fd)
"""
with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/test_sequence_split_indices.pickle","rb") as fr:
	train_seq_id_weight = pickle.load(fr)
	test_seq_id_weight = pickle.load(fr)

def cand_set_stat(process_num,num_init,num_query):
	print process_num, num_init, num_query
	try:
		cand_set_recall = []
		cand_set_overlap = []
		cand_set_cr = []
		cand_set_size_list = []
		cand_cov = 0.0
		cand_cr = 0.0
		cand_corr = set()
		cand_total = set()
		adop_total = set()

		l=0
		avg_num_adopters = 0
		for i in train_seq_id_weight:
			seq_sample_vocab = tag_seq[i]
			avg_num_adopters+=len(seq_sample_vocab)
			init_adopters=seq_sample_vocab[0:num_init]
			next_adopters = seq_sample_vocab[num_init:]
			N = num_query #1000 #M #num_users

			op, cand_set_size, cc, cr, M, med_qres, corr, cand, adop = get_candidate_set(init_adopters, next_adopters, N)
			cand_cov+=cc
			cand_cr+=cr
			cand_corr.update(corr)
			cand_total.update(cand)
			adop_total.update(adop)

			cand_set_recall.append(cc)
			cand_set_overlap.append(op)
			cand_set_cr.append(cr)
			cand_set_size_list.append(cand_set_size)

			print num_query, "cc", cc, "cand size", cand_set_size, "Avg", cand_cov*1./(l+1), sum(cand_set_size_list)*1./(l+1), "adop in cand", op, "total", M, "med qres size", med_qres, "uniq cand covered", len(cand_corr), "total adop", len(adop_total), "total cand", len(cand_total), l
			l+=1
			# if l%25==0:
			# 	print num_query, "cc", cc, "cr", cr, "cand size", cand_set_size, "Avg", cand_cov*1./(l+1), sum(cand_set_size_list)*1./(l+1), "adop in cand", op, "total", M, "med qres size", med_qres, l
			if l==train_ex_limit:
				break
		print process_num, num_init, num_query, "num examples", l, "cc", print_stats(cand_set_recall), avg_num_adopters*1./l, "uniq cand covered", len(cand_corr), "total adop", len(adop_total), "total cand", len(cand_total)
		# with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/candset_n"+str(num_init)+"_c"+str(num_query)+".pickle","wb") as fd:
		# # with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/candset_n"+str(num_init)+"_r"+str(num_query)+".pickle","wb") as fd:
		# 	pickle.dump(cand_set_recall,fd)
		# 	pickle.dump(cand_set_overlap,fd)
		# 	pickle.dump(cand_set_cr,fd)
		# 	pickle.dump(cand_set_size_list,fd)
	except Exception as e:
		print traceback.format_exc()

# tic = time.clock()
# cand_set_stat(0,num_init_adopters,top_k)
# toc = time.clock()
# print "cand set eval in", (toc-tic)*1000

num_workers = min(NUM_PROCESSES,cpu_count())
pool = Pool(processes=num_workers) 
process_num=0
for i in num_init_adopters:
	for j in top_k:
	# for j in query_rad:
		pool.apply_async(cand_set_stat, args=(process_num,i,j))
		process_num+=1
pool.close()
pool.join()

print vec_file, num_init_adopters, top_k