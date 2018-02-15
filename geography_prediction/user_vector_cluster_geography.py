#cluster user vectors using kmeans and get spread of geography, in terms of entropy, within each cluster and across clusters
#split user set into training and test for geography prediction using sklearn

import cPickle as pickle
import time
from math import sqrt, log
import numpy
# from scipy.spatial import cKDTree as KDTree
from sklearn.neighbors import NearestNeighbors
import sys
from sklearn.cluster import KMeans
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, precision_score, recall_score, accuracy_score
import random

vec_file = "/mnt/filer01/word2vec/node_vectors_1hr_pr.txt"
num_clusters_user_vec = 1000
tr = 0.1
user_sampling = True

print vec_file, tr, user_sampling

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
"""
# cluster user vectors
tic = time.clock()
kmeans = KMeans(n_clusters=num_clusters_user_vec, init='k-means++', n_init=5, max_iter=300, tol=0.0001, n_jobs=4)
vec_cluster_idx = list(kmeans.fit(vec).labels_)
toc = time.clock()
print "kmeans done in", (toc-tic)*1000
with open("/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/user_vector_cluster_index_1hr_pr.pickle","wb") as fd:
	pickle.dump(vec_cluster_idx,fd)
"""
# with open("/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/user_vector_cluster_index_1hr_pr.pickle","rb") as fr:
# 	vec_cluster_idx = pickle.load(fr)

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
# fetched_nodes = set()
map_id_not_found = 0
map_id_not_found_fr = 0
not_mapped_fr = 0
no_fol_id = 0
no_fr_id = 0

def getadj(node):
	# global adj
	# global fetched_nodes
	global map_id_not_found, no_fol_id
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
			followers.append(m[int(u[j])]) # check if u[j] in m
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

#entropy of geography distribution within clusters
"""
clusters = defaultdict(list)

for i in range(0,len(vocab)):
	clusters[vec_cluster_idx[i]].append(location_buckets[vocab[i]])

clusters_ent=[]
ent=0.0
for i in clusters:
	loc_dist = Counter(clusters[i])
	c_ent = 0.0
	for l in loc_dist:
		p = loc_dist[l]*1./len(clusters[i])
		if p > 0:
			c_ent+= -1.0*p*log(p,2)
	clusters_ent.append(c_ent)
	ent+=c_ent

print ent*1./len(clusters), min(clusters_ent), max(clusters_ent) 	

#entropy of geography distribution across clusters
loc_spread = defaultdict(list)
for i in range(0,len(vocab)):
	loc_spread[location_buckets[vocab[i]]].append(vec_cluster_idx[i])
loc_ent=[]
l_ent=0.0
for i in loc_spread:
	loc_dist = Counter(loc_spread[i])
	c_ent = 0.0
	for l in loc_dist:
		p = loc_dist[l]*1./len(loc_spread[i])
		if p > 0:
			c_ent+= -1.0*p*log(p,2)
	loc_ent.append(c_ent)
	l_ent+=c_ent

print l_ent*1./len(loc_spread), min(loc_ent), max(loc_ent), sorted(loc_ent)[len(loc_ent)/2] 

with open("/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/user_vector_cluster_entropy.pickle","wb") as fd:
	pickle.dump(clusters,fd)
	pickle.dump(clusters_ent,fd)
	pickle.dump(loc_spread,fd)
	pickle.dump(loc_ent,fd)
"""
#geography classification

def pred_eval(y_true,y_pred):
	macro_f1 = f1_score(y_true, y_pred, average='macro')
	micro_f1 = f1_score(y_true, y_pred, average='micro')
	acc = accuracy_score(y_true, y_pred, normalize = False)
	macro_prec = precision_score(y_true, y_pred, average='macro') 
	micro_prec = precision_score(y_true, y_pred, average='micro')
	macro_rec = recall_score(y_true, y_pred, average='macro') 
	micro_rec = recall_score(y_true, y_pred, average='micro') 
	return (macro_f1, micro_f1), acc, (macro_prec, micro_prec), (macro_rec, micro_rec)

#users with known location
X = []
Y = []
m_id = []

for i in range(0,len(vocab)):
	loc = location_buckets[vocab[i]]
	if loc!=-1:
		X.append(vec[i])
		Y.append(loc)
		m_id.append(vocab[i])

print len(X), len(vocab), len(X[0])

labelled_users = len(X)

idx = range(0,labelled_users)
random.shuffle(idx)
if user_sampling:
	idx = idx[0:1000000]

num_train = int(tr*len(idx))

X = numpy.asarray(X)
Y = numpy.asarray(Y)
m_id = numpy.asarray(m_id)

train_idx = idx[:num_train]
test_idx = idx[num_train:]

train_X, test_X = X[train_idx], X[test_idx]
train_Y, test_Y = Y[train_idx], Y[test_idx]
train_m_id, test_m_id = m_id[train_idx], m_id[test_idx]

train_m_id = set(train_m_id.tolist())
print len(train_X), len(test_X)

# train_label_freq = Counter(train_Y)
# print train_label_freq, len(train_label_freq)

# test_label_freq = Counter(test_Y)
# print test_label_freq, len(test_label_freq)

#majority classifier
maj_clf = DummyClassifier(strategy='most_frequent')
maj_clf.fit(train_X, train_Y)

maj_pred_Y = maj_clf.predict(test_X)
maj_label = maj_pred_Y[0]

print pred_eval(test_Y,maj_pred_Y), maj_label
# print Counter(maj_pred_Y)

#linear svm
svc_clf = LinearSVC(penalty='l2', C=10.0, dual=False, multi_class='ovr')
svc_clf.fit(train_X, train_Y)

svc_pred_Y = svc_clf.predict(test_X) 

print pred_eval(test_Y,svc_pred_Y)
svc_label_freq = Counter(svc_pred_Y)
print svc_label_freq, len(svc_label_freq)

#logistic regression
clf = LogisticRegression(penalty='l2', C=10.0, dual=False, solver='liblinear', multi_class='ovr')
clf.fit(train_X, train_Y)

pred_Y = clf.predict(test_X) 

print pred_eval(test_Y,pred_Y)
pred_label_freq = Counter(pred_Y)
print pred_label_freq, len(pred_label_freq)

#max followers
tic = time.clock()

fol_pred_Y = []
count = 0
not_pred = 0
limit_pred = []
test_Y_limit = []
for u in test_m_id:
	fol = getadj(u)
	known_fol = set(fol)&train_m_id
	if len(known_fol)==0:
		fol_pred_Y.append(maj_label)
		not_pred+=1
	else:
		loc_count = Counter([location_buckets[i] for i in known_fol])
		p = loc_count.most_common(1)[0][0]
		fol_pred_Y.append(p)
		limit_pred.append(p)
		test_Y_limit.append(test_Y[count])
	count+=1
	if count%100000==0:
		print "fol", count

toc = time.clock()
print "fol based pred", (toc-tic)*1000

print pred_eval(test_Y,fol_pred_Y)
freq = Counter(fol_pred_Y)
print freq, len(freq), not_pred
print "limit pred", pred_eval(test_Y_limit,limit_pred), "cov", 1 - not_pred*1./len(test_m_id)

#max friends
fr_pred_Y = []
count = 0
not_pred = 0
limit_pred = []
test_Y_limit = []
for u in test_m_id:
	fr = getadjfr(u)
	known_fr = set(fr)&train_m_id
	if len(known_fr)==0:
		fr_pred_Y.append(maj_label)
		not_pred+=1
	else:
		loc_count = Counter([location_buckets[i] for i in known_fr])
		p = loc_count.most_common(1)[0][0]
		fr_pred_Y.append(p)
		limit_pred.append(p)
		test_Y_limit.append(test_Y[count])
	count+=1
	if count%100000==0:
		print "fr", count

print pred_eval(test_Y,fr_pred_Y)
freq = Counter(fr_pred_Y)
print freq, len(freq), not_pred
print "limit pred", pred_eval(test_Y_limit,limit_pred), "cov", 1 - not_pred*1./len(test_m_id)

print "fol", map_id_not_found, no_fol_id, "fr", map_id_not_found_fr, not_mapped_fr, no_fr_id

# close file handles of follower list files
for f in f_read_list:
	f.close()
for f in f_read_list_fr:
	f.close()
print vec_file, tr, user_sampling