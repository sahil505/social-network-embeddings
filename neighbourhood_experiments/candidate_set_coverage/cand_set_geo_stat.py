#get nearest users to sample users using user vectors and write geography, friend and follower coverage stats
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
from collections import defaultdict

NUM_PROCESSES = 3

vec_file = "/mnt/filer01/word2vec/node_vectors_1hr_pr.txt"
adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt" #"sample_sequences"
# num_init_adopters = [10,20,50,100] #range(10,101,10)
user_sample_size = 1000
top_k = numpy.arange(0.9,1.15,0.05) #[1000,2000,5000,10000]# + range(2000,10001,2000)
query_rad = numpy.arange(0.6,1.05,0.05)
norm_vec = True

print vec_file, user_sample_size, top_k, query_rad, norm_vec

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
# arr = ["user_followers_bigger_graph.txt","user_followers_bigger_graph_2.txt", "user_followers_bigger_graph_i.txt","user_followers_bigger_graph_recrawl_2.txt", "user_followers_bigger_graph_recrawl_3.txt","user_followers_bigger_graph_recrawl.txt"]
# f_read_list = []
# for i in arr:
# 	f_read_list.append(open("/twitterSimulations/graph/" + i,'rb'))
	
# line_offset = pickle.load( open( "/twitterSimulations/follower_file_offset.pickle", "rb" ) )
# print 'Follower file offset Read\n'

# arr_fr = ["user_friends_bigger_graph.txt","user_friends_bigger_graph_2.txt", "user_friends_bigger_graph_i.txt","user_friends_bigger_graph_recrawl.txt"]
# f_read_list_fr = []
# for i in arr_fr:
# 	f_read_list_fr.append(open("/twitterSimulations/graph/" + i,'rb'))
	
# line_offset_fr = pickle.load( open( "friend_file_offset.pickle", "rb" ) )
# print 'Friend file offset Read\n'

# adj = dict() # check memory use, list of sets or list of lists
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

#read location files
location_buckets = [-1] * 7697889
# location_buckets = dict() #map to -1 for users not in location files
fr = open('/twitterSimulations/known_locations.txt', 'r')
for line in fr:
	line = line.rstrip()
	u = line.split('\t')
	try:
		location_buckets[m[int(u[0])]] = int(u[1])
	except:
		pass
fr.close()

#inferred location from followers
"""
fr = open('/twitterSimulations/known_locations1.txt', 'r')
for line in fr:
	line = line.rstrip()
	u = line.split('\t')
	try:
		location_buckets[m[int(u[0])]] = int(u[1])
	except:
		pass
fr.close()
print "location file read"
"""

#query candidate set
def get_candidate_set(query,followers,friends,N):
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

	cand_set = list(knn_list[0])#[1:]

	cand_set_size = len(cand_set)
	# if cand_set_size!=N:
	# 	print "error",len(knn_list[0]),len(cand_set)
	cand_set_users = [vocab[a] for a in cand_set]
	cand_set_users = set(cand_set_users)
	fol_overlap = len(cand_set_users&followers)
	fr_overlap = len(cand_set_users&friends)

	# print "candidate set recall", num_adopters, "out of", len(next_adopters), "cand size", len(cand_user_idx)
	return fol_overlap, fr_overlap, cand_set_size

#query candidate set and compare geography
def get_candidate_set_geo(query,geo,N):
	try:
		query_set_ind = vocab_index[query]
	except KeyError:
		print "query word not present"
		return
	query_vec = vec[query_set_ind]
	
	# query using scipy kdtree
	# d_list,knn_list = kd.query(query_vec,k=cand_size_factor*N+len(query_set_ind))
	
	# query using sklearn
	# knn_list = neigh.kneighbors(X=query_vec, n_neighbors=N+1, return_distance=False)
	
	# get vectors within distance N
	knn_list = neigh.radius_neighbors(X=query_vec, radius=N, return_distance=False)

	cand_set = list(knn_list[0])#[1:]

	cand_set_size = len(cand_set)-1
	# if cand_set_size!=N:
	# 	print "error",len(knn_list[0]),len(cand_set)
	geo_overlap = sum(location_buckets[vocab[u]]==geo for u in cand_set)-1

	# print "candidate set recall", num_adopters, "out of", len(next_adopters), "cand size", len(cand_user_idx)
	return geo_overlap, cand_set_size

def print_stats(u):
	# fol,fr = zip(*u)
	return [numpy.mean(u,axis=0), numpy.std(u,axis=0), numpy.median(u,axis=0)]

# sample users from vocab

known_loc_users = []
loc_size = defaultdict(int)
for i in vocab:
	loc = location_buckets[i]
	loc_size[loc]+=1
	if loc!=-1:
		known_loc_users.append(i)

print "known location for", len(known_loc_users), len(vocab)-len(known_loc_users)
# user_sample = random.sample(known_loc_users,user_sample_size)

user_sample = random.sample(vocab,user_sample_size)

def cand_cov_stat(process_num,num_query):
	print process_num, num_query
	try:
		cand_set_recall = []
		cand_set_overlap = []
		cand_set_size_list = []
		num_nbh = []
		fol_cov = 0.0
		fr_cov = 0.0
		geo_cov = 0.0
		geo_prec_cov = 0.0

		l=0
		for i in user_sample:
			# #followers
			# fol_t = getadj(i)
			# fol = set(fol_t)&vocab_set

			# #friends
			# fr_t = getadjfr(i)
			# fr = set(fr_t)&vocab_set

			#geography
			user_loc = location_buckets[i]
			user_loc_size = loc_size[user_loc]

			# num_nbh+=[(len(fol),len(fr))]
			num_nbh.append(user_loc_size)

			N = num_query
			# fol_op, fr_op, cand_set_size = get_candidate_set(i, fol, fr, N)
			geo_op, cand_set_size = get_candidate_set_geo(i, user_loc, N)

			# cc_fol = 0.0
			# if len(fol)!=0:
			# 	cc_fol = fol_op*1./len(fol)
			# fol_cov+=cc_fol

			# cc_fr = 0.0
			# if len(fr)!=0:
			# 	cc_fr = fr_op*1./len(fr)
			# fr_cov+=cc_fr

			cc_geo = geo_op*1./user_loc_size
			cc_geo_prec = 0.0
			if cand_set_size!=0:
				cc_geo_prec = geo_op*1./cand_set_size
			geo_cov+=cc_geo
			geo_prec_cov+=cc_geo_prec

			# cand_set_recall.append((cc_fol,cc_fr))
			# cand_set_overlap.append((fol_op,fr_op))
			# cand_set_size_list.append(cand_set_size)

			cand_set_recall.append((cc_geo,cc_geo_prec))
			cand_set_overlap.append(geo_op)
			cand_set_size_list.append(cand_set_size)

			l+=1
			# print num_query, "fol", cc_fol, "fr", cc_fr, "cand size", cand_set_size, "Avg", fol_cov*1./l, fr_cov*1./l, sum(cand_set_size_list)*1./l, "num in cand, total", (fol_op,len(fol)), (fr_op,len(fr)), l
			# print num_query, "geo cov", cc_geo, "prec", cc_geo_prec, "cand size", cand_set_size, "Avg", geo_cov*1./l, geo_prec_cov*1./l, sum(cand_set_size_list)*1./l, "num in cand, total", (geo_op,user_loc_size,cand_set_size), l
			if l%500==0:
				# print num_query, "fol", cc_fol, "fr", cc_fr, "cand size", cand_set_size, "Avg", fol_cov*1./l, fr_cov*1./l, sum(cand_set_size_list)*1./l, "num in cand, total", (fol_op,len(fol)), (fr_op,len(fr)), l
				print num_query, "geo cov", cc_geo, "prec", cc_geo_prec, "cand size", cand_set_size, "Avg", geo_cov*1./l, geo_prec_cov*1./l, sum(cand_set_size_list)*1./l, "num in cand, total", (geo_op,user_loc_size,cand_set_size), l
		print num_query, "num examples", l, "cc", print_stats(cand_set_recall), print_stats(num_nbh)
		# with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/candset_fol_fr_c"+str(num_query)+".pickle","wb") as fd:
		# with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/candset_loc_c"+str(num_query)+".pickle","wb") as fd:
		with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/candset_loc_r"+str(num_query)+"_check.pickle","wb") as fd:
			pickle.dump(cand_set_recall,fd)
			pickle.dump(cand_set_overlap,fd)
			pickle.dump(num_nbh,fd)
			pickle.dump(cand_set_size_list,fd)
	except Exception as e:
		print traceback.format_exc()

# tic = time.clock()
# cand_cov_stat(top_k)
# toc = time.clock()
# print "cand set eval in", (toc-tic)*1000

print "fol", map_id_not_found, no_fol_id, not_mapped_fol, "fr", map_id_not_found_fr, not_mapped_fr, no_fr_id

num_workers = min(NUM_PROCESSES,cpu_count())
pool = Pool(processes=num_workers) 
process_num=0
for i in top_k:
	pool.apply_async(cand_cov_stat, args=(process_num,i))
	process_num+=1
pool.close()
pool.join()

print vec_file, user_sample_size, top_k