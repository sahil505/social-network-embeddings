#get candidate users to the initial adopters of a hashtag sequence in test sequences using distance-based or nearest neighbour queries with user vectors,
#rank using model learned on distances of candidates from initial adopters and compare with actual adopters in the sequence
#train and test model on each topic in training using 5-fold cross validation, set c and r to get 0.3,0.4,0.5 cand set coverage

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
num_init_adopters = 20
par_m = 8
metric_Hausdorff_m_avg = 0
top_k = 1000
seq_len_threshold = 500
cand_size_factor = 1
train_ex_limit = 100
norm_vec = True
cv_fold = 5
top_k_test = 10

def init_clf():
	# clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, C=1.0, class_weight=None)
	# clf = SVC(C=1000.0, kernel='rbf', shrinking=True, probability=False, tol=0.001, cache_size=2000, class_weight=None, max_iter=-1)
	clf = RandomForestClassifier(n_estimators=300, n_jobs=4, class_weight=None)
	# clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, class_weight=None, max_iter=100)
	return clf

# training_options='-s 0 -t 2 -b 1 -m 8000'
print vec_file, num_init_adopters, top_k, train_ex_limit, top_k_test

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
vocab_set = set(vocab)
print "num users in train sequences", num_users
# print "users removed from vocab", len(set(users_train)-set(vocab))
# print "users in test sequences but not in vocab", len(users_test-set(vocab))

# building kd-tree
# tic = time.clock()
# # kd = KDTree(vec, leafsize=10)
# neigh = NearestNeighbors(n_neighbors=5, radius=1.0, algorithm='ball_tree', leaf_size=100, metric='minkowski', p=2) #'ball_tree', 'kd_tree', 'auto'
# neigh.fit(vec)
# toc = time.clock()
# print "ball tree built in", (toc-tic)*1000

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

def print_stats(res):
	# u,nb,f = zip(*res)
	# return [numpy.mean(u), numpy.std(u), numpy.median(u)],[numpy.mean(nb), numpy.std(nb), numpy.median(nb)],[numpy.mean(f), numpy.std(f), numpy.median(f)]
	nb,f = zip(*res)
	return [numpy.mean(nb), numpy.std(nb), numpy.median(nb)],[numpy.mean(f), numpy.std(f), numpy.median(f)]

#reading follower graph files
m = dict()
fr = open("/twitterSimulations/graph/map.txt")
for line in fr:
	line = line.rstrip()
	u = line.split(' ')
	m[int(u[0])] = int(u[1])
fr.close()
print 'Map Read'

arr = ["user_followers_bigger_graph.txt","user_followers_bigger_graph_2.txt", "user_followers_bigger_graph_i.txt","user_followers_bigger_graph_recrawl_2.txt", "user_followers_bigger_graph_recrawl_3.txt","user_followers_bigger_graph_recrawl.txt"]
f_read_list = []
for i in arr:
	f_read_list.append(open("/twitterSimulations/graph/" + i,'rb'))
	
line_offset = pickle.load( open( "/twitterSimulations/follower_file_offset.pickle", "rb" ) )
print 'Follower file offset Read\n'


adj = dict() # check memory use, list of sets or list of lists
fetched_nodes = set()
map_id_not_found = 0

def getadj(node):
	global adj
	global fetched_nodes
	global map_id_not_found
	if node in fetched_nodes:
		return adj[node]
	elif node in line_offset:
		adj[node] = set()
		followers = set()
		fetched_nodes.add(node) # fetched even if exits from an if loop
		(file_count, offset) = line_offset[node] # node is mapped id, check if node in line_offset
		f_read_list[file_count].seek(offset)
		line = f_read_list[file_count].readline()
		line = line.rstrip()
		u = line.split(' ')
		if(int(u[0]) > 7697889):
			print "Number of followers exceeded" #check, remove
			return None
		if len(u) <= 2:
			print "no follower list"
			return set()
		if m[int(u[1])]!=node:
			print "Error in index" #check, remove
			sys.exit(0) #check, remove
		for j in range(2,len(u)): # get two-hops list also
			followers.add(m[int(u[j])]) # check if u[j] in m
			#adj[node].add(m[int(u[j])])
		followers = followers&vocab_set
		adj[node].update(followers)
		return followers
	else:
		adj[node] = set()
		fetched_nodes.add(node) # fetched even if exits from an if loop
		map_id_not_found+=1
		print "offset not found", node #check, remove
		return set()

def get_Nranked_list_fol(query_set,N):
	friend_count = dict()
	init_adopters = query_set
	sec_hop = 1 #2
	while (sec_hop>0):
		for a in init_adopters:
			followers = getadj(a)
			for f in followers-set(query_set):
				try:
					friend_count[f]+=1
				except KeyError:
					friend_count[f]=1
		init_adopters = friend_count.keys()
		sec_hop-=1
	friend_count_list = [(f,friend_count[f]) for f in friend_count]
	ranked_list = [f for f,_ in nlargest(N,friend_count_list,key=lambda x: x[1])]
	if len(friend_count_list)>=N:
		return ranked_list
	else:
		users_left = N-len(friend_count_list)
		print "followers ranked list short", users_left
		for i in nb_seq_order:
			if i not in friend_count and i not in query_set:
				ranked_list.append(i)
				users_left-=1
			if users_left==0:
				break
		return ranked_list

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
nb_seq_order = pickle.load(open(nb_sorted_pickle,"rb"))
print len(nb_seq_order)
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
	ap_total = []
	prec_k_total = []
	rec_k_total = []

	mean_ap_nbapp=0.0
	mean_prec_k_nbapp=0.0
	mean_rec_k_nbapp=0.0

	mean_ap_fol=0.0
	mean_prec_k_fol=0.0
	mean_rec_k_fol=0.0

	l=0
	avg_num_adopters = 0
	for i in train_seq_id_weight:
		
		seq_sample_vocab = tag_seq[i]
		avg_num_adopters+=len(seq_sample_vocab)
		init_adopters=seq_sample_vocab[0:num_init]
		next_adopters = seq_sample_vocab[num_init:]
		next_adopters = set(next_adopters)
		M = len(next_adopters)
		N = num_query #1000 #M #num_users

		#number of hashtags adopted in training, count baseline
		nb = []
		nb_num = 0
		for u in nb_seq_order:
			if u not in init_adopters:
				nb.append(u)
				nb_num+=1
			if nb_num==N:
				break
		precision_k_nbapp = 0.0
		num_hits_nbapp = 0.0
		for k,p in enumerate(nb):
			if p in next_adopters:
				num_hits_nbapp+=1.0
				precision_k_nbapp += num_hits_nbapp/(k+1.0)
		average_precision_nbapp = precision_k_nbapp/min(M,N)
		# prec_r_nbapp = num_hits_nbapp/M
		prec_k_nbapp = num_hits_nbapp/N
		rec_k_nbapp = num_hits_nbapp/M
		mean_ap_nbapp+=average_precision_nbapp
		# mean_prec_r_nbapp+=prec_r_nbapp
		mean_prec_k_nbapp+=prec_k_nbapp
		mean_rec_k_nbapp+=rec_k_nbapp
		
		#follower of adopters baseline
		fol_seq_order = get_Nranked_list_fol(init_adopters,N)
		precision_k_fol = 0.0
		num_hits_fol = 0.0
		for k,p in enumerate(fol_seq_order):
			if p in next_adopters:
				num_hits_fol+=1.0
				precision_k_fol += num_hits_fol/(k+1.0)
		average_precision_fol = precision_k_fol/min(M,N)
		# prec_r_fol = num_hits_fol/M
		prec_k_fol = num_hits_fol/N
		rec_k_fol = num_hits_fol/M
		# print "Nb_App", "RPrecision", prec_r_fol
		mean_ap_fol+=average_precision_fol
		# mean_prec_r_fol+=prec_r_fol
		mean_prec_k_fol+=prec_k_fol
		mean_rec_k_fol+=rec_k_fol
		#, "MRP", mean_prec_r_fol/float(l+1)
		
		ap_total.append((average_precision_nbapp,average_precision_fol))
		# prec_r_total.append((
		prec_k_total.append((prec_k_nbapp,prec_k_fol))
		rec_k_total.append((rec_k_nbapp,rec_k_fol))
		
		print num_init, num_query, "Nb", "ap", average_precision_nbapp, "prec", prec_k_nbapp, "rec", rec_k_nbapp, "MAP", mean_ap_nbapp/float(l+1), "MPk", mean_prec_k_nbapp/float(l+1), "MRk", mean_rec_k_nbapp/float(l+1), "Fol", "ap", average_precision_fol, "prec", prec_k_fol, "rec", rec_k_fol, "MAP", mean_ap_fol/float(l+1), "MPk", mean_prec_k_fol/float(l+1), "MRk", mean_rec_k_fol/float(l+1), M, l
		l+=1
		# if l%25==0:
		# 	print num_init, num_query, "Nb", "ap", average_precision_nbapp, "prec", prec_k_nbapp, "rec", rec_k_nbapp, "MAP", mean_ap_nbapp/float(l+1), "MPk", mean_prec_k_nbapp/float(l+1), "MRk", mean_rec_k_nbapp/float(l+1), "Fol", "ap", average_precision_fol, "prec", prec_k_fol, "rec", rec_k_fol, "MAP", mean_ap_fol/float(l+1), "MPk", mean_prec_k_fol/float(l+1), "MRk", mean_rec_k_fol/float(l+1), M, l
		if l==train_ex_limit:
			break
	
	print num_init, num_query, "MAP", print_stats(ap_total), "MPk", print_stats(prec_k_total), "MRk", print_stats(rec_k_total), avg_num_adopters*1./l
	# with open("adopter_pred_files/single_topic_train_test_files/baseline_eval_n"+str(num_init)+"_rf_prec"+str(num_query)+".pickle","wb") as fd:
	# 	pickle.dump(prec_k_total,fd)
	# 	pickle.dump(rec_k_total,fd)
	# 	pickle.dump(ap_total,fd)

adop_pred_stat(0,num_init_adopters,top_k_test)

print vec_file, num_init_adopters, top_k, top_k_test