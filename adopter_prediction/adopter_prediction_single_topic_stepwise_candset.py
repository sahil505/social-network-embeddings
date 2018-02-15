#get candidate users to the initial adopters of a hashtag sequence in test sequences using distance-based or nearest neighbour queries with user vectors,
#rank using model learned on distances of candidates from initial adopters and compare with actual adopters in the sequence
#train and test model on each topic in training using 5-fold cross validation, add predicted users into query set stepwise
#and train on distances of candidates from bigger query set, set c and r to get 0.3,0.4,0.5 cand set coverage

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
from multiprocessing import Pool, cpu_count
import traceback

NUM_PROCESSES = 10

vec_file = "/mnt/filer01/word2vec/node_vectors_1hr_pr.txt"
nb_sorted_pickle = "/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/baseline_user_order_1hr_pr.pickle"
adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt" #"sample_sequences"
num_init_adopters = [10,20,30,40,50]
par_m = 8
metric_Hausdorff_m_avg = 0
top_k = [[4000,15000],[2500,10500],[1500,7000],[1000,6000],[750,4500]]
# top_k = [[.955,1.05],[.81,.91,1.],[.76,.86,.95],[.75,.85,.925],[.725,.84,.915]]
seq_len_threshold = 500
cand_size_factor = 1
train_ex_limit = 100
norm_vec = True
cv_fold = 5
top_k_test = 10
next_k = 2

def init_clf():
	# clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, C=1.0, class_weight=None)
	# clf = SVC(C=1000.0, kernel='rbf', shrinking=True, probability=False, tol=0.001, cache_size=2000, class_weight=None, max_iter=-1)
	clf = RandomForestClassifier(n_estimators=300, n_jobs=4, class_weight=None)
	# clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, class_weight=None, max_iter=100)
	return clf

# training_options='-s 0 -t 2 -b 1 -m 8000'
print vec_file, num_init_adopters, top_k, train_ex_limit, top_k_test, next_k

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

def get_cand_feature_vectors(query_set,next_adopters,N):
	try:
		query_set_ind = [ vocab_index[query] for query in query_set ]
	except KeyError:
		print "query word not present"
		return
	query_vec = [vec[i] for i in query_set_ind]
	# query using scipy kdtree
	# _,knn_list = kd.query(query_vec,k=cand_size_factor*N+len(query_set_ind))
	# query using sklearn
	_,knn_list = neigh.kneighbors(X=query_vec, n_neighbors=cand_size_factor*N+len(query_set_ind), return_distance=True)
	# get vectors within distance N
	# _,knn_list = neigh.radius_neighbors(X=query_vec, radius=N, return_distance=True)

	cand_set = set()
	for index_list in knn_list:
		filtered=[idx for idx in index_list if idx not in query_set_ind]
		cand_set.update(filtered)

	M = len(next_adopters)
	next_adopters_index = [vocab_index[a] for a in next_adopters]
	next_adopters_index = set(next_adopters_index)

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
		label=-1
		if idx in next_adopters_index:
			label=1
			num_adopters+=1
		X.append(dist_query_set)
		Y.append(label)
		cand_user.append(idx)

	cand_set_size = len(cand_user)
	# print "candidate set recall", num_adopters, "out of", M, "cand size", cand_set_size
	cr = 0.0
	if cand_set_size!=0:
		cr = num_adopters*1./cand_set_size
	cc = 0.0
	if M!=0:
		cc = num_adopters*1./M
	return X,Y,cand_user,cc,cr,num_adopters

def get_dist_vector(X,cand_user,pred_adopters):
	X_exp = []
	query_vec = [vec[i] for i in pred_adopters]
	for count,idx in enumerate(cand_user):
		dist_vec = X[count]
		dist_query_set = [0.0]*len(pred_adopters)
		cand_vec = vec[idx]
		l=0
		for q in query_vec:
			dist = sum( (cand_vec[x]-q[x])**2 for x in xrange(0,dim) )
			dist_query_set[l]= sqrt(dist)
			l+=1
		# avg = sum(dist_query_set)*1./l
		dist_vec+=dist_query_set
		dist_vec=sorted(dist_vec)
		# dist_vec.append(avg)
		X_exp.append(dist_vec)
	return X_exp

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
with open("/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/sequence_file_split_indices_weight_n40.pickle","wb") as fd:
	pickle.dump(train_seq_id_weight,fd)
	pickle.dump(test_seq_id_weight,fd)
"""
#sequence indices from test set with atleast 500 adopters
with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/test_sequence_split_indices.pickle","rb") as fr:
	train_seq_id_weight = pickle.load(fr)
	test_seq_id_weight = pickle.load(fr)

def adop_pred_stat(process_num,num_init,num_query):
	print process_num, num_init, num_query
	try:
		prec_k_total = []
		rec_k_total = []
		cand_set_recall = []
		cand_set_cr = []
		cand_set_size_list = []
		cand_set_overlap = []
		cand_cov = 0.0
		cand_cr = 0.0
		avg_num_adopters = 0

		l=0
		for i in train_seq_id_weight:
			
			seq_sample_vocab = tag_seq[i]
			avg_num_adopters+=len(seq_sample_vocab)
			init_adopters=seq_sample_vocab[0:num_init]
			next_adopters = seq_sample_vocab[num_init:]
			M = len(next_adopters)
			N = num_query #1000 #M #num_users

			X, Y, cand_user_idx, cc, cr, op = get_cand_feature_vectors(init_adopters, next_adopters, N)
			cand_cov+=cc
			cand_cr+=cr

			X = numpy.asarray(X)
			Y = numpy.asarray(Y)
			cand_user_idx = numpy.asarray(cand_user_idx)

			# with open("adopter_pred_files/single_topic_train_test_files/train_file_n10_"+str(l)+".pickle","wb") as fd:
			# 	pickle.dump(X,fd)
			# 	pickle.dump(Y,fd)
			"""
			with open("adopter_pred_files/single_topic_train_test_files/train_file_n10_"+str(l)+".pickle","rb") as fr:
				X = pickle.load(fr)
				Y = pickle.load(fr)
			cc=0
			cr=0
			"""
			cand_set_size = len(X)

			precision_k = 0.0
			recall_k = 0.0
			#cross-validation, random split
			kf = KFold(cand_set_size, n_folds=cv_fold, shuffle=True)
			for train_ind,test_ind in kf:
				train_X, test_X = X[train_ind], X[test_ind]
				train_Y, test_Y = Y[train_ind], Y[test_ind]

				test_adopt_ind = [ind for ind in test_ind if Y[ind]==1]
				num_adopt = len(test_adopt_ind)

				train_X_new = train_X.tolist()
				train_cand_user = cand_user_idx[train_ind]
				test_X_new = test_X.tolist()
				test_ind_new = test_ind
				pred_adopters_ind_total = []
				# print len(train_X_new[0]), len(train_X_new), len(test_ind_new)
				for s in range(next_k,top_k_test+1,next_k):

					#re-initialise
					clf_t = init_clf()
					clf_t.fit(train_X_new, train_Y)

					# p_vals_adopt = clf_t.decision_function(test_X_new)
					p_vals = clf_t.predict_proba(test_X_new)
					try:
						cls_ind = list(clf_t.classes_).index(1)
						p_vals_adopt = [p[cls_ind] for p in p_vals]
					except ValueError:
						p_vals_adopt = [0.0 for p in p_vals]

					cand_prob_list = zip(test_ind_new,p_vals_adopt)
					
					#precision at k
					pred_adopters_ind = [w for w,_ in nlargest(next_k,cand_prob_list,key=lambda x: x[1])]
					pred_adopters = cand_user_idx[pred_adopters_ind]
					pred_adopters_ind_total+=pred_adopters_ind

					if s<top_k_test:
						test_new_enum = [(i,x) for i,x in enumerate(test_ind_new) if x not in pred_adopters_ind]
						test_cv_ind_new, test_ind_new = zip(*test_new_enum)
						test_cv_ind_new = list(test_cv_ind_new)
						test_ind_new = list(test_ind_new)
						test_cand_user = cand_user_idx[test_ind_new]
						test_X_new_temp = [test_X_new[i] for i in test_cv_ind_new]
						test_X_new = get_dist_vector(test_X_new_temp,test_cand_user,pred_adopters)
						train_X_new = get_dist_vector(train_X_new,train_cand_user,pred_adopters)
					# print s, len(train_X_new[0]), len(train_X_new), len(pred_adopters_ind), len(pred_adopters_ind_total), len(set(pred_adopters_ind_total)), len(test_ind_new), len(test_X_new[0]), len(test_X_new), len(set(pred_adopters_ind+test_ind_new))

				num_hits = len(set(test_adopt_ind)&set(pred_adopters_ind_total))
				# print len(pred_adopters_ind_total), top_k_test
				prec_k_cv = num_hits*1./top_k_test
				if num_adopt!=0:
					rec_k_cv = num_hits*1./num_adopt
				else:
					rec_k_cv = 0

				#precision at R
				# pred_adopters = [w for w,_ in nlargest(num_adopt,cand_prob_list,key=lambda x: x[1])]
				# if num_adopt!=0:
				# 	prec_k_cv = len(set(test_adopt_ind)&set(pred_adopters))*1./num_adopt
				# else:
				# 	prec_k_cv = 0

				# print "precision", prec_k_cv, "recall", rec_k_cv, "num adopters", num_adopt, len(test_Y)
				precision_k += prec_k_cv
				recall_k += rec_k_cv

			precision_k = precision_k*1./cv_fold
			recall_k = recall_k*1./cv_fold
			prec_k_total.append(precision_k)
			rec_k_total.append(recall_k)
			cand_set_recall.append(cc)
			cand_set_cr.append(cr)
			cand_set_size_list.append(cand_set_size)
			cand_set_overlap.append(op)

			# print num_init, num_query, "Avg precision", precision_k, "Avg recall", recall_k, "precision total", sum(prec_k_total)*1./(l+1), "recall total", sum(rec_k_total)*1./(l+1), "op", op, "in", cand_set_size, "from", M, "cc", cc, "cr", cr, l
			l+=1
			if l%25==0:
				print num_init, num_query, "Avg precision", precision_k, "Avg recall", recall_k, "precision total", sum(prec_k_total)*1./(l+1), "recall total", sum(rec_k_total)*1./(l+1), "op", op, "in", cand_set_size, "from", M, "cc", cc, "cr", cr, l
			if l==train_ex_limit:
				break
		print num_init, num_query, "num examples", l, "avg cand set recall", cand_cov*1./l, "avg cand set cr", cand_cr*1./l, "avg cand set size", numpy.mean(cand_set_size_list), "avg adopt", avg_num_adopters*1./l

		print num_init, num_query, "Precision", print_stats(prec_k_total), "Recall", print_stats(rec_k_total)
		with open("adopter_pred_files/single_topic_train_test_files/eval_n"+str(num_init)+"_c"+str(num_query)+"_rf_prec"+str(top_k_test)+"_step"+str(next_k)+".pickle","wb") as fd:
		# with open("adopter_pred_files/single_topic_train_test_files/eval_n"+str(num_init)+"_r"+str(num_query)+"_rf_prec"+str(top_k_test)+"_step"+str(next_k)+".pickle","wb") as fd:
			pickle.dump(prec_k_total,fd)
			pickle.dump(rec_k_total,fd)
			pickle.dump(cand_set_recall,fd)
			pickle.dump(cand_set_cr,fd)
			pickle.dump(cand_set_size_list,fd)
			pickle.dump(cand_set_overlap,fd)
	except Exception as e:
		print traceback.format_exc()

# tic = time.clock()
# adop_pred_stat(0,num_init_adopters,top_k)
# toc = time.clock()
# print "cand set eval in", (toc-tic)*1000

num_workers = min(NUM_PROCESSES,cpu_count())
pool = Pool(processes=num_workers) 
process_num=0
for i,q in zip(num_init_adopters,top_k):
	for j in q:
		pool.apply_async(adop_pred_stat, args=(process_num,i,j))
		process_num+=1
pool.close()
pool.join()

print vec_file, num_init_adopters, top_k, top_k_test, next_k