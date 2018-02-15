#split user set into training and test for activity time prediction using sklearn

import cPickle as pickle
import time, datetime
from dateutil import tz
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
adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt" #"sample_sequences"
num_clusters_user_vec = 1000
tr = 0.5

print vec_file, tr

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

with open("/mnt/filer01/word2vec/degree_distribution/sequence_file_split_indices.pickle","rb") as fr:
	_ = pickle.load(fr)
	test_seq_id = pickle.load(fr)
test_seq_id = set(test_seq_id)

#activity time classification

uneven_bins = [(0,5),(6,8),(9,11),(12,14),(15,17),(18,20),(21,23)]
num_bins = len(uneven_bins)
def get_time_bin(hr):
	c=0
	for i,j in uneven_bins:
		if hr>=i and hr<=j:
			return c
		c+=1

def pred_eval(y_true,y_pred):
	macro_f1 = f1_score(y_true, y_pred, average='macro')
	micro_f1 = f1_score(y_true, y_pred, average='micro')
	acc = accuracy_score(y_true, y_pred, normalize = False)
	macro_prec = precision_score(y_true, y_pred, average='macro') 
	micro_prec = precision_score(y_true, y_pred, average='micro')
	macro_rec = recall_score(y_true, y_pred, average='macro') 
	micro_rec = recall_score(y_true, y_pred, average='micro') 
	return (macro_f1, micro_f1), acc, (macro_prec, micro_prec), (macro_rec, micro_rec)

#users in most frequent location

selected_users = set()
for i in range(0,len(vocab)):
	loc = location_buckets[vocab[i]]
	if loc==6:
		selected_users.add(vocab[i])

#prominent slots for users
tweet_time = defaultdict(list)
count=0
# nb_seq = dict()
# adlen = []
utc_to_dt = datetime.datetime.utcfromtimestamp
present_tz = tz.gettz('UTC')
target_tz = tz.gettz('America/New_York')
with open(adoption_sequence_filename, "rb") as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		# if count in test_seq_id:
		for i in xrange(1, len(u)):
			timestamp = int(u[i][0:u[i].index(',')])
			author = int(u[i][u[i].index(',')+1 : ])
			if author in selected_users:
				hr = utc_to_dt(timestamp).replace(tzinfo=present_tz).astimezone(target_tz).hour
				slot = get_time_bin(hr) #hr/3
				tweet_time[author].append(slot)
		count+=1
print "seq considered", count

total_activity_bins = [0]*num_bins
total_tweets = 0
time_slots = dict()
count=0
sample_activity = dict()
num_tweets_avg = []
for a in tweet_time:
	bins=[0]*num_bins
	user_tweet_times = tweet_time[a]
	num_tweets_avg.append(len(user_tweet_times))
	bin_count = Counter(user_tweet_times)
	p = bin_count.most_common(1)[0][0]
	for t in bin_count:
		freq = bin_count[t]
		total_activity_bins[t]+=freq
		total_tweets+=freq
	time_slots[a]=p
	if len(sample_activity)<1000:
		sample_activity[a]=bin_count
	count+=1
	if count%50000==0:
		print "time slot done", count

# with open("/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/activity_time_uneven_bins_testset.pickle","wb") as fd:
# 	pickle.dump(total_activity_bins,fd)
# 	pickle.dump(total_tweets,fd)
# with open("/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/sample_user_activity_time_uneven_bins.pickle","wb") as fd:
# 	pickle.dump(sample_activity,fd)

X = []
Y = []
m_id = []

not_found = 0
for a in selected_users:
	if a in time_slots:
		X.append(vec[vocab_index[a]])
		Y.append(time_slots[a])
		m_id.append(a)
	else:
		not_found+=1

X = numpy.asarray(X)
Y = numpy.asarray(Y)
m_id = numpy.asarray(m_id)

print len(X), len(selected_users), len(vocab), len(X[0]), not_found, [numpy.mean(num_tweets_avg), numpy.std(num_tweets_avg), numpy.median(num_tweets_avg)]

labelled_users = len(X)
idx = range(0,labelled_users)
random.shuffle(idx)
num_train = int(tr*len(idx))

train_idx = idx[:num_train]
test_idx = idx[num_train:]

train_X, test_X = X[train_idx], X[test_idx]
train_Y, test_Y = Y[train_idx], Y[test_idx]
train_m_id, test_m_id = m_id[train_idx], m_id[test_idx]

train_m_id = set(train_m_id.tolist())
print len(train_X), len(test_X)

train_label_freq = Counter(train_Y)
print train_label_freq, len(train_label_freq)

test_label_freq = Counter(test_Y)
print test_label_freq, len(test_label_freq)

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
		loc_count = Counter([time_slots[i] for i in known_fol])
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
		loc_count = Counter([time_slots[i] for i in known_fr])
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
print vec_file, tr