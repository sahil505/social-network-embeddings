#get nearest users to a fraction of adopters of a hashtag sequence in test sequences using user vectors and compare with followers of the adopters

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
frac_adopters = 0.5
train_ex_limit = 50
norm_vec = True
seq_len_threshold = 500 #top_k
cand_size_factor = 3
# num_init_adopters = 10
# top_k = 100 #range(4000,10001,2000)
# query_rad = 0.8

print vec_file, frac_adopters, train_ex_limit, seq_len_threshold, cand_size_factor, norm_vec

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

# adj = dict() # check memory use, list of sets or list of lists
adj = [[]] * 7697889

for i in range(0, 7697889):
	adj[i] = []

fetched_nodes = set()
map_id_not_found = 0
not_mapped_fol = 0
no_fol_id = 0

def getadj(node):
	global adj
	global fetched_nodes
	global map_id_not_found, no_fol_id, not_mapped_fol
	if node in fetched_nodes:
		return adj[node]
	if node in line_offset:
		# adj[node] = []
		followers = []
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
		adj[node]=followers
		return followers
	else:
		# adj[node] = []
		fetched_nodes.add(node) # fetched even if exits from an if loop
		map_id_not_found+=1
		# print "offset not found", node #check, remove
		return []

def get_candidate_set(query_set,N,total_cand):
	try:
		query_set_ind = [ vocab_index[query] for query in query_set ]
	except KeyError:
		print "query word not present"
		return
	query_vec = [vec[i] for i in query_set_ind]
		
	#?use distance_upper_bound for set_size queries sequentially
	# d_list,knn_list = kd.query(query_vec,k=N+len(query_set_ind)) #, eps=eps)
	d_list,knn_list = neigh.kneighbors(X=query_vec, n_neighbors=cand_size_factor*N, return_distance=True)

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
		if count==total_cand:
			break
	return knn
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

def get_Jaccard_index(a,b):
	ind = 0.0
	if len(b)!=0:
		ind = len(a&b)*1./len(a|b)
	return ind

def compare_nbr_set():
	cand_set_recall = []
	cand_set_overlap = []
	cand_set_size_list = []
	cand_cov_fol = 0.0
	cand_cov_vec = 0.0
	cand_size_differ = 0
	jaccard_ind = 0.0

	l=0
	avg_num_adopters = 0
	for i in train_seq_id_weight:
		seq_sample_vocab = tag_seq[i]
		seq_len = len(seq_sample_vocab)
		avg_num_adopters+=seq_len
		num_init = int(float(frac_adopters)*seq_len)
		init_adopters=seq_sample_vocab[0:num_init]
		next_adopters = seq_sample_vocab[num_init:]
		next_adopters = set(next_adopters)
		num_next_adopters = len(next_adopters)

		#followers of adopters
		fol_cand = set()
		for a in init_adopters:
			fol_cand.update(set(getadj(a))&vocab_set)
		fol_cand = fol_cand - set(init_adopters)
		num_adopt_fol = len(fol_cand&next_adopters)
		cc_fol = num_adopt_fol*1./num_next_adopters

		#neighbours of user vectors
		cand_set_size_fol = len(fol_cand)
		N = int(cand_set_size_fol*1./num_init)+num_init #cand_set_size_fol
		user_vec_cand = get_candidate_set(init_adopters, N, cand_set_size_fol)
		cand_set_size = len(user_vec_cand)
		if cand_set_size!=cand_set_size_fol:
			cand_size_differ+=1
		user_vec_cand = set(user_vec_cand)
		num_adopt_vec = len(user_vec_cand&next_adopters)
		cc_vec = num_adopt_vec*1./num_next_adopters

		cand_cov_fol+=cc_fol
		cand_cov_vec+=cc_vec

		# jaccard_ind = get_Jaccard_index(fol_cand,user_vec_cand)

		cand_set_recall.append((cc_fol,cc_vec))
		cand_set_overlap.append((num_adopt_fol,num_adopt_vec))
		cand_set_size_list.append((len(fol_cand),cand_set_size,jaccard_ind))

		print frac_adopters, "adopters taken", num_init, "cc", cc_fol, cc_vec, "sim", jaccard_ind, "cand size", cand_set_size_fol, cand_set_size, "Avg", cand_cov_fol*1./(l+1), cand_cov_vec*1./(l+1), numpy.mean(cand_set_size_list,axis=0), "adop in cand", num_adopt_fol, num_adopt_vec, "total", seq_len, l
		l+=1
		# if l%25==0:
		# 	print num_query, "cc", cc, "cr", cr, "cand size", cand_set_size, "Avg", cand_cov*1./(l+1), sum(cand_set_size_list)*1./(l+1), "adop in cand", op, "total", M, "med qres size", med_qres, l
		if l==train_ex_limit:
			break
	print frac_adopters, "num examples", l, "cc", print_stats(cand_set_recall), avg_num_adopters*1./l, cand_size_differ
	with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/nbr_frac"+str(frac_adopters)+"_seq"+str(seq_len_threshold)+"_tr50.pickle","wb") as fd:
		pickle.dump(cand_set_recall,fd)
		pickle.dump(cand_set_overlap,fd)
		pickle.dump(cand_set_size_list,fd)

tic = time.clock()
compare_nbr_set()
toc = time.clock()
print "nbr set eval in", (toc-tic)*1000
print "fol", map_id_not_found, no_fol_id, not_mapped_fol

# close file handles of follower list files
for f in f_read_list:
	f.close()

print vec_file, frac_adopters, train_ex_limit, seq_len_threshold, cand_size_factor, norm_vec