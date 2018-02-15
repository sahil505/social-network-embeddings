#get neighbourhood of a sample of users using user vectors and compare with network neighbours

import cPickle as pickle
import time
from math import sqrt
import random
from heapq import nsmallest, nlargest, merge
import numpy
# from scipy.spatial import cKDTree as KDTree
from sklearn.neighbors import NearestNeighbors
import sys

vec_file = "/mnt/filer01/word2vec/node_vectors_1hr_pr.txt"
adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt" #"sample_sequences"
user_sample_size = 10000
frac_adopters = 0.50
train_ex_limit = 50
norm_vec = True
seq_len_threshold = 500 #top_k
cand_size_factor = 10
# num_init_adopters = 10
# top_k = 100 #range(4000,10001,2000)
# query_rad = 0.8

print vec_file, frac_adopters, train_ex_limit, seq_len_threshold, user_sample_size, cand_size_factor, norm_vec

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
tic = time.clock()
# kd = KDTree(vec, leafsize=10)
neigh = NearestNeighbors(n_neighbors=5, radius=1.0, algorithm='ball_tree', leaf_size=100, metric='minkowski', p=2) #'ball_tree', 'kd_tree', 'auto'
neigh.fit(vec)
toc = time.clock()
print "ball tree built in", (toc-tic)*1000

m = dict()
fr = open("/twitterSimulations/graph/map.txt")
for line in fr:
	line = line.rstrip()
	u = line.split(' ')
	m[int(u[0])] = int(u[1])
fr.close()
print 'Map Read'

#reading follower graph files
arr = ["user_followers_bigger_graph.txt","user_followers_bigger_graph_2.txt", "user_followers_bigger_graph_i.txt","user_followers_bigger_graph_recrawl_2.txt", "user_followers_bigger_graph_recrawl_3.txt","user_followers_bigger_graph_recrawl.txt"]
f_read_list = []
for i in arr:
	f_read_list.append(open("/twitterSimulations/graph/" + i,'rb'))
	
line_offset = pickle.load( open( "/twitterSimulations/follower_file_offset.pickle", "rb" ) )
print 'Follower file offset Read\n'

arr_fr = ["user_friends_bigger_graph.txt","user_friends_bigger_graph_2.txt", "user_friends_bigger_graph_i.txt","user_friends_bigger_graph_recrawl.txt"]
f_read_list_fr = []
for i in arr_fr:
	f_read_list_fr.append(open("/twitterSimulations/graph/" + i,'rb'))
	
line_offset_fr = pickle.load( open( "friend_file_offset.pickle", "rb" ) )
print 'Friend file offset Read\n'

# adj = dict() # check memory use, list of sets or list of lists
# adj = [[]] * 7697889

# for i in range(0, 7697889):
# 	adj[i] = []

# fetched_nodes = set()
map_id_not_found = 0
map_id_not_found_fr = 0
not_mapped_fol = 0
not_mapped_fr = 0
no_fol_id = 0
no_fr_id = 0

def getadj(node):
	# global adj
	# global fetched_nodes
	global map_id_not_found, no_fol_id, not_mapped_fol
	# if node in fetched_nodes:
	# 	return adj[node]
	if node in line_offset:
		# adj[node] = []
		followers = []
		# fetched_nodes.add(node) # fetched even if exits from an if loop
		(file_count, offset) = line_offset[node] # node is mapped id, check if node in line_offset
		f_read_list[file_count].seek(offset)
		line = f_read_list[file_count].readline()
		line = line.rstrip()
		u = line.split(' ')
		if(int(u[0]) > 7697889):
			print "Number of followers exceeded" #check, remove
			return None
		if len(u) <= 2:
			# print "no follower list"
			no_fol_id+=1
			return []
		if m[int(u[1])]!=node:
			print "Error in index" #check, remove
			sys.exit(0) #check, remove
		for j in range(2,len(u)): # get two-hops list also
			fol = int(u[j])
			if fol in m:
				followers.append(m[fol]) # check if u[j] in m
			else:
				not_mapped_fol+=1
			#adj[node].add(m[int(u[j])])
		# adj[node]=followers
		return followers
	else:
		# adj[node] = []
		# fetched_nodes.add(node) # fetched even if exits from an if loop
		map_id_not_found+=1
		# print "offset not found", node #check, remove
		return []

#get friend adjacency list
def getadjfr(node):
	# global adj
	# global fetched_nodes
	global map_id_not_found_fr, not_mapped_fr, no_fr_id
	# if node in fetched_nodes:
	# 	return adj[node]
	if node in line_offset_fr:
		# adj[node] = []
		friends = []
		# fetched_nodes.add(node) # fetched even if exits from an if loop
		(file_count, offset) = line_offset_fr[node] # node is mapped id, check if node in line_offset_fr
		f_read_list_fr[file_count].seek(offset)
		line = f_read_list_fr[file_count].readline()
		line = line.rstrip()
		u = line.split(' ')
		if(int(u[0]) > 7697889):
			print "Number of friends exceeded" #check, remove
			return None
		if len(u) <= 2:
			# print "no friend list"
			no_fr_id+=1
			return []
		if m[int(u[1])]!=node:
			print "Error in friend index" #check, remove
			sys.exit(0) #check, remove
		for j in range(2,len(u)): # get two-hops list also
			fr = int(u[j])
			if fr in m:
				friends.append(m[fr]) # check if u[j] in m
			else:
				not_mapped_fr+=1
			#adj[node].add(m[int(u[j])])
		# adj[node]=friends
		return friends
	else:
		# adj[node] = []
		# fetched_nodes.add(node) # fetched even if exits from an if loop
		map_id_not_found_fr+=1
		# print "offset not found, friend index", node #check, remove
		return []

#query candidate set
def get_candidate_set(query,N):
	try:
		query_set_ind = vocab_index[query]
	except KeyError:
		print "query word not present"
		return
	query_vec = vec[query_set_ind]
	
	# query using scipy kdtree
	# d_list,knn_list = kd.query(query_vec,k=cand_size_factor*N+len(query_set_ind))
	
	# query using sklearn
	knn_list = neigh.kneighbors(X=query_vec, n_neighbors=N+1, return_distance=False)
	
	# get vectors within distance N
	# knn_list = neigh.radius_neighbors(X=query_vec, radius=N, return_distance=False)

	cand_set = list(knn_list[0])[1:]

	cand_set_size = len(cand_set)
	if cand_set_size!=N:
		print "error",len(knn_list[0]),len(cand_set)
	cand_set_users = [vocab[a] for a in cand_set]

	# print "candidate set recall", num_adopters, "out of", len(next_adopters), "cand size", len(cand_user_idx)
	return cand_set_users

"""
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
	return num_adopters, cand_set_size, cc, cr, M, med_qresult_size
"""

def print_stats(u):
	return [numpy.mean(u,axis=0), numpy.std(u,axis=0), numpy.median(u,axis=0)]

# reading test sequences
not_found_vocab=[]
# source_thr = 1395858601 + 12*60*60
# non_emergent_tags = pickle.load(open("/mnt/filer01/word2vec/degree_distribution/nonEmergentHashtags.pickle","rb"))
tag_seq = []
count=0
test_seq_users = set()
# nb_seq = dict()
# adlen = []
with open(adoption_sequence_filename, "rb") as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		not_found = set()
		# first_timestamp = int(u[1][0:u[1].index(',')])
		# first tweet only after source_thr timestamp
		# if first_timestamp>=source_thr
		# check if <5 tweets in 12 hours for emergent hashtags, not already popular
		# u[0] not in non_emergent_tags and
		if count in test_seq_id:
			adopters = set()
			for i in xrange(1, len(u)):
				#timestamp = int(u[i][0:u[i].index(',')])
				author = int(u[i][u[i].index(',')+1 : ])
				if author in vocab_index:
					# removing repeat adopters
					adopters.add(author)
				else:
					not_found.add(author)
			if len(adopters)>0:
				tag_seq.append(adopters)
				test_seq_users.update(adopters)
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

#train-test split for learning weights
num_train = int(0.5*len(seq_index_filter))
print "training examples present", num_train
train_seq_id_weight = seq_index_filter[:num_train]
test_seq_id_weight = seq_index_filter[num_train:]
with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/test_sequence_split_indices_seq"+str(seq_len_threshold)+".pickle","wb") as fd:
	pickle.dump(train_seq_id_weight,fd)
	pickle.dump(test_seq_id_weight,fd)
"""
with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/test_sequence_split_indices_seq"+str(seq_len_threshold)+".pickle","rb") as fr:
	train_seq_id_weight = pickle.load(fr)
	test_seq_id_weight = pickle.load(fr)

test_seq_id_total = train_seq_id_weight+test_seq_id_weight
print len(test_seq_id_total), len(test_seq_users)
user_sample = random.sample(vocab,user_sample_size)

def get_Jaccard_index(a,b):
	ind = 0.0
	if len(b)!=0:
		ind = len(a&b)*1./len(a|b)
	return ind

def compare_nbr_set():
	cand_set_recall = []
	cand_set_overlap = []
	cand_set_size_list = []
	# ind_fol = 0.0
	# ind_fr = 0.0

	l=0
	not_found_fol = 0
	not_found_tag = 0
	not_found_adopt = 0
	for i in user_sample:

		# seq_sample_vocab = tag_seq[i]
		# seq_len = len(seq_sample_vocab)
		# avg_num_adopters+=seq_len
		# num_init = int(float(frac_adopters)*seq_len)
		# init_adopters=seq_sample_vocab[0:num_init]
		# next_adopters = seq_sample_vocab[num_init:]
		# num_next_adopters = len(next_adopters)
		# next_adopters = set(next_adopters)

		#followers, friends
		fol = set(getadj(i))&vocab_set
		# fr = set(getadjfr(i))&vocab_set

		# if len(fol)==0 and len(fr)==0:
		# 	not_found+=1
		# 	continue

		#neighbours of user vectors
		cand_set_size_fol = len(fol) #max(len(fol),len(fr))
		user_vec_nbh = set(get_candidate_set(i, cand_set_size_fol))
		cand_set_size = len(user_vec_nbh)
		if cand_set_size_fol!=cand_set_size:
			print "error in cand set size"
		fol_adopt = 0.0
		vec_adopt = 0.0
		num_tags = 0
		cc_fol = 0.0
		cc_vec = 0.0

		if cand_set_size_fol==0:
			not_found_fol+=1
			continue
		if i in test_seq_users:
			for j in test_seq_id_total:
				adopters = tag_seq[j]
				if i in adopters:
					fol_adopt += len(adopters&fol)
					vec_adopt += len(adopters&user_vec_nbh)
					num_tags += 1
			if num_tags!=0:
				cc_fol = fol_adopt*1./num_tags
				cc_vec = vec_adopt*1./num_tags
			else:
				not_found_tag+=1
		else:
			not_found_adopt+=1
		# vec_fol = get_Jaccard_index(set(user_vec_nbh[0:len(fol)]),fol)
		# vec_fr = get_Jaccard_index(set(user_vec_nbh[0:len(fr)]),fr)

		# ind_fol+=vec_fol
		# ind_fr+=vec_fr

		# cand_set_recall.append((vec_fol,vec_fr))
		# cand_set_size_list.append((len(fol),len(fr),cand_set_size))

		cand_set_recall.append((cc_fol,cc_vec))
		cand_set_overlap.append((fol_adopt,vec_adopt))
		cand_set_size_list.append((cand_set_size,num_tags))

		l+=1
		if l%50==0:
			print "cc", cc_fol, cc_vec, "op", fol_adopt, vec_adopt, "num tags", num_tags, "cand size", cand_set_size, "Avg", numpy.mean(cand_set_overlap,axis=0), numpy.mean(cand_set_size_list,axis=0), not_found_fol, not_found_tag, not_found_adopt, l
		# if l==train_ex_limit:
		# 	break
	print "num examples", l, "cc", print_stats(cand_set_recall), print_stats(cand_set_overlap), not_found_fol, not_found_tag, not_found_adopt
	with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/nbr_fol_fr_coadopt"+str(user_sample_size)+".pickle","wb") as fd:
		pickle.dump(cand_set_recall,fd)
		pickle.dump(cand_set_overlap,fd)
		pickle.dump(cand_set_size_list,fd)

tic = time.clock()
compare_nbr_set()
toc = time.clock()
print "nbr set eval in", (toc-tic)*1000
print "fol", map_id_not_found, no_fol_id, not_mapped_fol, "fr", map_id_not_found_fr, not_mapped_fr, no_fr_id

# close file handles of follower list files
for f in f_read_list:
	f.close()
for f in f_read_list_fr:
	f.close()

print vec_file, user_sample_size