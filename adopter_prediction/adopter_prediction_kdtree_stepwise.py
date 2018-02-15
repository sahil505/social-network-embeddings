#get nearest users to the initial adopters of a hashtag sequence in test sequences using user vectors and compare with actual adopters in the sequence
#query users stepwise starting from initial adopters and adding next k nearest users consecutively

import cPickle as pickle
import time
from math import sqrt
import random
from heapq import nsmallest, nlargest, merge
import numpy
# from scipy.spatial import cKDTree as KDTree
import sys
from sklearn.neighbors import NearestNeighbors

vec_file = "/mnt/filer01/word2vec/node_vectors_1hr_pr.txt"
nb_sorted_pickle = "/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/baseline_user_order_1hr_pr.pickle"
adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt" #"sample_sequences"
num_init_adopters = 30
par_m = 8
metric_Hausdorff_m_avg = 2
rankedlist_method = 3 #0-stepwise_pred,1-min,2-avg, 3-avg in cand set, 4 - min from centroid
top_k = 10
seq_len_threshold = 500 #top_k
seq_count_limit=100
norm_vec = True
next_k = 2
print vec_file, num_init_adopters, metric_Hausdorff_m_avg, top_k, norm_vec, rankedlist_method, next_k

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
print "kdtree tree built in", (toc-tic)*1000

def get_Nranked_list_kdtree(query_set,N,from_centroid):
	try:
		query_set_ind = [ vocab_index[query] for query in query_set ]
	except KeyError:
		print "query word not present"
		return
	query_vec = [vec[i] for i in query_set_ind]
	if from_centroid:
		cent = [float(sum(i))/len(i) for i in zip(*query_vec)]
		query_vec = [cent]
		
	#?use distance_upper_bound for set_size queries sequentially
	# d_list,knn_list = kd.query(query_vec,k=N+len(query_set_ind)) #, eps=eps)
	d_list,knn_list = neigh.kneighbors(X=query_vec, n_neighbors=N+len(query_set_ind), return_distance=True)

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

#not correct, points not in candidate set can be in top-N
def get_Nranked_list_kdtree_avg(query_set,N):
	# print query_set
	try:
		query_set_ind = [ vocab_index[query] for query in query_set ]
	except KeyError:
		print "query word not present"
		return
	query_vec = [vec[i] for i in query_set_ind]
	#?use distance_upper_bound for set_size queries sequentially
	# d_list,knn_list = kd.query(query_vec,k=N+len(query_set_ind))
	d_list,knn_list = neigh.kneighbors(X=query_vec, n_neighbors=N+len(query_set_ind), return_distance=True)

	# print d_list,knn_list
	cand_set = set()
	for index_list in knn_list:
		filtered=[idx for idx in index_list if idx not in query_set_ind]
		cand_set.update(filtered)
	# print cand_set
	cand_user=[]
	for idx in cand_set:
		# print idx, vocab[idx]
		dist_query_set = [0.0]*len(query_set)
		cand_vec = vec[idx]
		l=0
		for q in query_vec:
			dist = sum( (cand_vec[x]-q[x])**2 for x in xrange(0,dim) )
			dist_query_set[l]= sqrt(dist)
			l+=1
		# print dist_query_set
		cand_user.append((vocab[idx],sum(dist_query_set)*1./l))
	# print cand_user
	knn_avg = [w for w,_ in nsmallest(N,cand_user,key=lambda x: x[1])]
	return knn_avg

def get_Nranked_list(query_set,N):
	# wordN = [0]*N
	# distN = [0.0]*N
	dist_total = []
	set_size = len(query_set)
	try:
		query_set_ind = [ vocab_index[query] for query in query_set ]
	except KeyError:
		print "query word not present"
		return
	for i in xrange(0,len(vec)):
		if i in query_set_ind:
			continue
		pres_word = vocab[i]
		pres_vec = vec[i]
		dist_k = [0.0]*set_size
		k=0
		for voc_ind in query_set_ind:
			user_vec = vec[voc_ind]
			#Euclidean distance (user_vec[x]-pres_vec[x])**2, same as cosine dis-similarity user_vec[x]*pres_vec[x] for norm 1 when subtracted by 1 and multiplied by 2
			dist = sum( (user_vec[x]-pres_vec[x])**2 for x in xrange(0,dim) )
			dist_k[k]= sqrt(dist)
			k+=1
			# dist = 0.0
			# for x in xrange(0,dim):
			# 	dist+=(user_vec[x]-pres_vec[x])**2 
		#distance of a point from a set
		# dist_k_sorted = sorted(dist_k)
		nearest_k = min(dist_k) # dist_k_sorted[0] #  if sorted not needed
		if metric_Hausdorff_m_avg==1:
			if nearest_k!=0.0:
				dist_set=sum( (nearest_k/dist_k[p])**(par_m) for p in xrange(0,set_size) )
				dist_set = nearest_k * (dist_set)**(1.0/set_size)
			else:
				dist_set=0.0
		elif metric_Hausdorff_m_avg==2:
			dist_set = sum(dist_k)*1./set_size
		else:
			dist_set=nearest_k
		dist_total.append((pres_word,dist_set))
		# for j in xrange(0,N):
		# 	if dist>distN[j]:
		# 		for k in xrange(N-1,j,-1):
		# 			distN[k] = distN[k-1]
		# 			wordN[k] = wordN[k-1]
		# 		distN[j] = dist
		# 		wordN[j] = pres_word
		# 		break
	# wordN = [w for w,_ in nsmallest(N,dist_total,key=lambda x: x[1])]
	wordN,distN = zip(*nsmallest(N,dist_total,key=lambda x: x[1]))
	# print "brute", distN
	return list(wordN) #zip(wordN,distN)

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

num_seqs=0
mean_ap=0.0
# mean_prec_r=0.0
mean_prec_k=0.0
mean_rec_k=0.0
mean_ap_nbapp=0.0
# mean_prec_r_nbapp=0.0
mean_prec_k_nbapp=0.0
mean_rec_k_nbapp=0.0
mean_ap_fol=0.0
# mean_prec_r_fol=0.0
mean_prec_k_fol=0.0
mean_rec_k_fol=0.0
ap_total = []
# prec_r_total = []
prec_k_total = []
rec_k_total = []
avg_num_adopters = 0

#test sequences in random order
# seq_random_index=range(0,len(tag_seq))
# random.shuffle(seq_random_index)

#sequence indices from test set with atleast 500 adopters
with open("/mnt/filer01/word2vec/degree_distribution/candset_stat_files/test_sequence_split_indices.pickle","rb") as fr:
	train_seq_id_weight = pickle.load(fr)
	test_seq_id_weight = pickle.load(fr)

# for i in seq_random_index:
for i in train_seq_id_weight:
	seq_sample_vocab = tag_seq[i]
	avg_num_adopters+=len(seq_sample_vocab)
	init_adopters=seq_sample_vocab[0:num_init_adopters]
	seq_sample_vocab = set(seq_sample_vocab[num_init_adopters:])
	M = len(seq_sample_vocab)
	N = top_k #1000 #M #num_users
	# if M<seq_len_threshold:
	# 	continue
	not_found=not_found_vocab[i]
	num_seqs+=1
	
	#precision, recall evaluation, user vectors
	if rankedlist_method==0:
		adopters_vec = []
		while len(adopters_vec)<N:
			seed_adopters = init_adopters + adopters_vec
			adopters_vec += get_Nranked_list_kdtree(seed_adopters,next_k,False)
		adopters_vec = adopters_vec[:N]
	elif rankedlist_method==1:
		adopters_vec = get_Nranked_list_kdtree(init_adopters,N,False)
	elif rankedlist_method==2:
		adopters_vec = get_Nranked_list(init_adopters,N)
	elif rankedlist_method==3:
		adopters_vec = get_Nranked_list_kdtree_avg(init_adopters,N)
	else:
		adopters_vec = get_Nranked_list_kdtree(init_adopters,N,True)
	# adopters_vec_kdtree = get_Nranked_list_kdtree_avg(init_adopters,N)
	# if adopters_vec!=adopters_vec_kdtree:
	# 	print "not same points", "same", len(set(adopters_vec)&set(adopters_vec_kdtree)), "out of", N
	# else:
	# 	print "same"
	# print [vocab_index[u] for u in adopters_vec_kdtree]
	# print [vocab_index[u] for u in adopters_vec]
	# continue
	precision_k = 0.0
	num_hits = 0.0
	for k,p in enumerate(adopters_vec):
		if p in seq_sample_vocab:
			num_hits+=1.0
			precision_k += num_hits/(k+1.0)
	average_precision = precision_k/min(M,N)
	# prec_r = num_hits/M
	prec_k = num_hits/N
	rec_k = num_hits/M
	print "Avg precision", average_precision, "Precision", prec_k, "Recall", rec_k, "users not found", not_found, "adopters in seq", M
	# print "RPrecision", prec_r
	mean_ap+=average_precision
	# mean_prec_r+=prec_r
	mean_prec_k+=prec_k
	mean_rec_k+=rec_k
	print "User vec", "MAP", mean_ap/float(num_seqs), "MPk", mean_prec_k/float(num_seqs), "MRk", mean_rec_k/float(num_seqs)
	#, "MRP", mean_prec_r/float(num_seqs)
	
	#number of hashtags adopted in training baseline
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
		if p in seq_sample_vocab:
			num_hits_nbapp+=1.0
			precision_k_nbapp += num_hits_nbapp/(k+1.0)
	average_precision_nbapp = precision_k_nbapp/min(M,N)
	# prec_r_nbapp = num_hits_nbapp/M
	prec_k_nbapp = num_hits_nbapp/N
	rec_k_nbapp = num_hits_nbapp/M
	print "Nb_App", "Avg precision", average_precision_nbapp, "Precision", prec_k_nbapp, "Recall", rec_k_nbapp
	# print "Nb_App", "RPrecision", prec_r_nbapp
	mean_ap_nbapp+=average_precision_nbapp
	# mean_prec_r_nbapp+=prec_r_nbapp
	mean_prec_k_nbapp+=prec_k_nbapp
	mean_rec_k_nbapp+=rec_k_nbapp
	print "Nb_App", "MAP", mean_ap_nbapp/float(num_seqs), "MPk", mean_prec_k_nbapp/float(num_seqs), "MRk", mean_rec_k_nbapp/float(num_seqs)
	#, "MRP", mean_prec_r_nbapp/float(num_seqs)
	
	#follower of adopters baseline
	fol_seq_order = get_Nranked_list_fol(init_adopters,N)
	precision_k_fol = 0.0
	num_hits_fol = 0.0
	for k,p in enumerate(fol_seq_order):
		if p in seq_sample_vocab:
			num_hits_fol+=1.0
			precision_k_fol += num_hits_fol/(k+1.0)
	average_precision_fol = precision_k_fol/min(M,N)
	# prec_r_fol = num_hits_fol/M
	prec_k_fol = num_hits_fol/N
	rec_k_fol = num_hits_fol/M
	print "Fol", "Avg precision", average_precision_fol, "Precision", prec_k_fol, "Recall", rec_k_fol
	# print "Nb_App", "RPrecision", prec_r_fol
	mean_ap_fol+=average_precision_fol
	# mean_prec_r_fol+=prec_r_fol
	mean_prec_k_fol+=prec_k_fol
	mean_rec_k_fol+=rec_k_fol
	print "Fol", "MAP", mean_ap_fol/float(num_seqs), "MPk", mean_prec_k_fol/float(num_seqs), "MRk", mean_rec_k_fol/float(num_seqs), "num seq", num_seqs
	#, "MRP", mean_prec_r_fol/float(num_seqs)
	
	ap_total.append((average_precision,average_precision_nbapp,average_precision_fol))
	# prec_r_total.append((
	prec_k_total.append((prec_k,prec_k_nbapp,prec_k_fol))
	rec_k_total.append((rec_k,rec_k_nbapp,rec_k_fol))

	seq_count_limit-=1
	if seq_count_limit==0:
		break
print "number of seq considered", num_seqs, "avg adopt", avg_num_adopters*1./num_seqs
print "MAP", "user vectors", mean_ap/float(num_seqs), "Nb_App", mean_ap_nbapp/float(num_seqs), "Fol", mean_ap_fol/float(num_seqs)
# print "MRP", mean_prec_r/float(num_seqs), mean_prec_r_nbapp/float(num_seqs), mean_prec_r_fol/float(num_seqs)
print "MPk", mean_prec_k/float(num_seqs), mean_prec_k_nbapp/float(num_seqs), mean_prec_k_fol/float(num_seqs), "MRk", mean_rec_k/float(num_seqs), mean_rec_k_nbapp/float(num_seqs), mean_rec_k_fol/float(num_seqs)
#pickle.dump(source_time,open("source_time.pickle","wb"))
def print_stats(res):
	u,nb,f = zip(*res)
	return [numpy.mean(u), numpy.std(u), numpy.median(u)],[numpy.mean(nb), numpy.std(nb), numpy.median(nb)],[numpy.mean(f), numpy.std(f), numpy.median(f)]
print "MAP", print_stats(ap_total)
print "MPk", print_stats(prec_k_total)
print "MRk", print_stats(rec_k_total)
with open("adopter_pred_files/acc_min_avg_stats/eval_n"+str(num_init_adopters)+"_prec"+str(top_k)+"_rank"+str(rankedlist_method)+"_pr_d100.pickle","wb") as fd:
	pickle.dump(ap_total,fd)
	pickle.dump(prec_k_total,fd)
	pickle.dump(rec_k_total,fd)
print vec_file, num_init_adopters, rankedlist_method, metric_Hausdorff_m_avg, top_k, norm_vec