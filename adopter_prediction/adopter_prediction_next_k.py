#get nearest users to the initial adopters of a hashtag sequence in test sequences using user vectors,
#rank based on learned weighted sum of distances of candidates from initial adopters and compare with actual adopters in the sequence
#predict next k adopters for each topic in training using increasing number of initial adopters and minimum distance metric

import cPickle as pickle
import time
from math import sqrt
import random
from heapq import nsmallest, nlargest, merge
import numpy
# from scipy.spatial import cKDTree as KDTree
from sklearn.neighbors import NearestNeighbors
import sys
# sys.path.append('libsvm-3.20/python')
# from svmutil import *
# sys.path.append('liblinear-1.96/python')
# from liblinearutil import *
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold

vec_file = "/mnt/filer01/word2vec/node_vectors_1hr_pr.txt"
nb_sorted_pickle = "/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/baseline_user_order_1hr_pr.pickle"
adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt" #"sample_sequences"
num_init_adopters = 10
par_m = 8
metric_Hausdorff_m_avg = 0
top_k = 100
seq_len_threshold = top_k #500
cand_size_factor = 1
train_ex_limit = 100
norm_vec = True
next_k_test = 10
next_k_limit = 100

def init_clf():
	# clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, C=1.0, class_weight=None)
	# clf = SVC(C=1000.0, kernel='rbf', shrinking=True, probability=False, tol=0.001, cache_size=2000, class_weight=None, max_iter=-1)
	clf = RandomForestClassifier(n_estimators=300, n_jobs=10, class_weight=None)
	# clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, class_weight=None, max_iter=100)
	return clf

# training_options='-s 0 -t 2 -b 1 -m 8000'
print vec_file, num_init_adopters, metric_Hausdorff_m_avg, top_k, cand_size_factor, train_ex_limit, next_k_test

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
			vocab.append(word)
			vectors.append(vec_norm)
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
print "kdtree tree built in", (toc-tic)*1000

def get_Nranked_list_kdtree(query_set,N):
	try:
		query_set_ind = [ vocab_index[query] for query in query_set ]
	except KeyError:
		print "query word not present"
		return
	query_vec = [vec[i] for i in query_set_ind]
	#?use distance_upper_bound for set_size queries sequentially
	# query using scipy kdtree
	# d_list,knn_list = kd.query(query_vec,k=N+len(query_set_ind)) #, eps=eps)
	# query using sklearn
	d_list,knn_list = neigh.kneighbors(X=query_vec, n_neighbors=N+len(query_set_ind), return_distance=True)
	#?use heap of size set_size and push top elements from set_size ranked list until N elements are popped
	index_dist_list = []
	for d,index in zip(d_list,knn_list):
		filtered=[(dt,idx) for (dt,idx) in list(zip(d,index)) if idx not in query_set_ind]
		index_dist_list.append(filtered)
	knn=[]
	sel=set()
	count=0
	for (d,idx) in merge(*index_dist_list):
		if idx not in sel:
			sel.add(idx)
			knn.append(vocab[idx])
			count+=1
		if count==N:
			break
	return knn

def get_cand_feature_vectors(query_set,next_adopters,N):
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

	X=[]
	Y=[]
	cand_user=[]
	num_adopters = 0
	for idx in cand_set:
		dist_query_set = [0.0]*len(query_set)
		cand_vec = vec[idx]
		l=0
		for q in query_vec:
			dist = sum( (cand_vec[x]-q[x])**2 for x in xrange(0,dim) )
			dist_query_set[l]= sqrt(dist)
			l+=1
		# avg = sum(dist_query_set)*1./l
		dist_query_set=sorted(dist_query_set)
		# dist_query_set.append(avg)
		user_m_id = vocab[idx]
		label=-1
		if user_m_id in next_adopters:
			label=1
			num_adopters+=1
		X.append(dist_query_set)
		Y.append(label)
		cand_user.append(user_m_id)
	print "candidate set recall", num_adopters, "out of", len(next_adopters), "cand size", len(cand_user)
	cr = num_adopters*1./len(cand_user)
	cc = 0.0
	if len(next_adopters)!=0:
		cc = num_adopters*1./len(next_adopters)
	return X,Y,cand_user,cc,cr

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
			if len(seq)>num_init_adopters:
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

prec_k_total = []
rec_k_total = []

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
with open("adopter_pred_files/sequence_file_split_indices_weight_n40.pickle","wb") as fd:
	pickle.dump(train_seq_id_weight,fd)
	pickle.dump(test_seq_id_weight,fd)
"""
with open("/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/sequence_file_split_indices_weight_n10.pickle","rb") as fr:
	train_seq_id_weight = pickle.load(fr)
	test_seq_id_weight = pickle.load(fr)

# train_X = []
# train_Y = []
l=0
for i in train_seq_id_weight:
	
	seq_sample_vocab = tag_seq[i]

	precision_k = []
	recall_k = []

	#continuously increase initial adopters for training
	for init_size in range(num_init_adopters,next_k_limit,next_k_test):	
		init_adopters=seq_sample_vocab[0:init_size]
		next_k_adopters = set(seq_sample_vocab[init_size:init_size+next_k_test])
		M = len(next_k_adopters)
		N = top_k #1000 #M #num_users

		adopters_vec = get_Nranked_list_kdtree(init_adopters,N)
		
		num_hits = len(set(next_k_adopters)&set(adopters_vec))
		prec_k_cv = num_hits*1./N
		rec_k_cv = num_hits*1./M

		#precision at R
		# pred_adopters = [w for w,_ in nlargest(M,cand_prob_list,key=lambda x: x[1])]
		# prec_k_cv = len(set(next_k_adopters)&set(pred_adopters))*1./M

		print "precision", prec_k_cv, "recall", rec_k_cv, "num adopters", M
		precision_k.append(prec_k_cv)
		recall_k.append(rec_k_cv)

	prec_k_total.append(precision_k)
	rec_k_total.append(recall_k)
	
	print "Avg precision", precision_k, "Avg recall", recall_k, "precision total", numpy.mean(prec_k_total, axis=0), "recall total", numpy.mean(rec_k_total, axis=0)
	l+=1
	if l%20==0:
		print "example num", l
	if l==train_ex_limit:
		break
print "training examples taken", l

def print_stats(u):
	return [numpy.mean(u, axis=0), numpy.std(u, axis=0), numpy.median(u, axis=0)]
print "Precision", print_stats(prec_k_total), "Recall", print_stats(rec_k_total)
with open("adopter_pred_files/single_topic_next_k/eval_n10_min_k10.pickle","wb") as fd:
	pickle.dump(prec_k_total,fd)
	pickle.dump(rec_k_total,fd)
print vec_file, num_init_adopters, metric_Hausdorff_m_avg, top_k