#visualising users in the candidate set of initial adopters for a particular hashtag along with predictions of single topic models, using both Random Forest and Logistic Regression models, using t-SNE or pca on user vectors

import matplotlib
matplotlib.use('Agg')
from tsne import *
import cPickle as pickle
import time
from math import sqrt
import random
from heapq import nsmallest, nlargest, merge
import numpy
# from scipy.spatial import cKDTree as KDTree
from sklearn.neighbors import NearestNeighbors
import sys

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold

vec_file = "/mnt/filer01/word2vec/node_vectors_1hr_pr.txt"
nb_sorted_pickle = "/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/baseline_user_order_1hr_pr.pickle"
adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt" #"sample_sequences"
num_init_adopters = 10
top_k = 1000
seq_len_threshold = top_k #500
cand_size_factor = 1
train_ex_limit = 100
norm_vec = True
cv_fold = 5
top_k_test = 100
use_tsne = True

def init_clf():
	# clf = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, C=1.0, class_weight=None)
	# clf = SVC(C=1000.0, kernel='rbf', shrinking=True, probability=False, tol=0.001, cache_size=2000, class_weight=None, max_iter=-1)
	clf = RandomForestClassifier(n_estimators=300, n_jobs=10, class_weight=None)
	# clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, class_weight=None, max_iter=100)
	return clf
	
def init_clf_logistic():
	clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, class_weight=None, max_iter=100)
	return clf

# training_options='-s 0 -t 2 -b 1 -m 8000'
print vec_file, num_init_adopters, top_k, cand_size_factor, top_k_test, train_ex_limit

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
print "kdtree tree built in", (toc-tic)*1000

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
	cc = 0.0
	if len(next_adopters)!=0:
		cc = num_adopters*1./len(next_adopters)
	return X,Y,cand_user,cc

# reading test sequences
not_found_vocab=[]
# source_thr = 1395858601 + 12*60*60
# non_emergent_tags = pickle.load(open("/mnt/filer01/word2vec/degree_distribution/nonEmergentHashtags.pickle","rb"))

tag_seq = []
tag_name = []
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
				tag_name.append(u[0])
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

cand_cov = 0.0
prec_k_total = 0.0
prec_k_total_log = 0.0

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

# with open("/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/train_file_weight_c1_n10.pickle","rb") as fr:
# 	train_X = pickle.load(fr)
# 	train_Y = pickle.load(fr)
# clf = init_clf()
# clf.fit(train_X, train_Y)
# print clf.get_params()

l=0
tag_val = []
for i in train_seq_id_weight:
	
	tag = tag_name[i]
	seq_sample_vocab = tag_seq[i]
	init_adopters=seq_sample_vocab[0:num_init_adopters]
	seq_sample_vocab = set(seq_sample_vocab[num_init_adopters:])
	M = len(seq_sample_vocab)
	N = top_k #1000 #M #num_users

	X, Y, cand_user, cc = get_cand_feature_vectors(init_adopters, seq_sample_vocab, N)
	cand_cov+=cc

	X = numpy.asarray(X)
	Y = numpy.asarray(Y)
	predY = [-1]*len(Y)
	predY_log = [-1]*len(Y)
	# cand_user = numpy.asarray(cand_user)
	cand_set_size = len(X)
	
	precision_k = 0.0
	precision_k_log = 0.0
	#cross-validation, random split
	kf = KFold(cand_set_size, n_folds=cv_fold, shuffle=True)
	for train_ind,test_ind in kf:
		train_X, test_X = X[train_ind], X[test_ind]
		train_Y, test_Y = Y[train_ind], Y[test_ind]

		# test_cand_user = cand_user[test_ind]
		
		test_adopt_ind = [ind for ind,val in enumerate(test_Y) if val==1]
		num_adopt = len(test_adopt_ind)
		#re-initialise
		clf_t = init_clf()
		clf_t.fit(train_X, train_Y)

		# p_vals_adopt = clf_t.decision_function(test_X)
		p_vals = clf_t.predict_proba(test_X)
		p_vals_adopt = [p[1] for p in p_vals]

		cand_prob_list = zip(range(0,len(test_X)),p_vals_adopt)
		
		#precision at k
		pred_adopters = [w for w,_ in nlargest(top_k_test,cand_prob_list,key=lambda x: x[1])]
		for ind in pred_adopters:
			predY[test_ind[ind]] = 1
		
		prec_k_cv = len(set(test_adopt_ind)&set(pred_adopters))*1./top_k_test
		
		#logistic
		clf_t_log = init_clf_logistic()
		clf_t_log.fit(train_X, train_Y)
		p_vals_log = clf_t_log.predict_proba(test_X)
		p_vals_adopt_log = [p[1] for p in p_vals_log]
		cand_prob_list_log = zip(range(0,len(test_X)),p_vals_adopt_log)
		#precision at k
		pred_adopters_log = [w for w,_ in nlargest(top_k_test,cand_prob_list_log,key=lambda x: x[1])]
		for ind in pred_adopters_log:
			predY_log[test_ind[ind]] = 1
		prec_k_cv_log = len(set(test_adopt_ind)&set(pred_adopters_log))*1./top_k_test
		
		print "precision", prec_k_cv, prec_k_cv_log,"num adopters in test", num_adopt, len(test_Y)
		precision_k += prec_k_cv
		precision_k_log += prec_k_cv_log

	precision_k = precision_k*1./cv_fold
	precision_k_log = precision_k_log*1./cv_fold
	prec_k_total += precision_k
	prec_k_total_log += precision_k_log

	print "Avg precision", precision_k, precision_k_log, "precision total", prec_k_total*1./(l+1), prec_k_total_log*1./(l+1), "cc", cc
	"""
	with open("/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/single_topic_vis/train_file_n10_"+str(l)+".pickle","wb") as fd:
		pickle.dump(X,fd)
		pickle.dump(Y,fd)
		pickle.dump(predY,fd)
		pickle.dump(cand_user,fd)
		pickle.dump(init_adopters,fd)
		pickle.dump(precision_k,fd)
		pickle.dump(tag,fd)
	"""
	"""
	with open("/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/single_topic_vis/train_file_n10_"+str(l)+".pickle","rb") as fr:
		X = pickle.load(fr)
		Y = pickle.load(fr)
		predY = pickle.load(fr)
		cand_user = pickle.load(fr)
		init_adopters = pickle.load(fr)
		precision_k = pickle.load(fr)
		tag = pickle.load(fr)
	"""
	tag_val.append((init_adopters,zip(cand_user,Y,predY,predY_log),tag,precision_k,precision_k_log))

	l+=1
	if l%20==0:
		print "example num", l
	if l==train_ex_limit:
		break
print "training examples taken", l, "avg candidate set recall", cand_cov*1./l

vec_limit = 1000
def get_user_vectors(t):
	init_adopters,cand,tag,prec,prec_log = tag_val[t]
	vectors = []
	color = []
	color_log = []
	count=0
	for u in init_adopters:
		vectors.append(vec[vocab_index[u]])
		color.append(0)
		color_log.append(0)
	random.shuffle(cand)
	for u,y,yp,yp_log in cand:
		vectors.append(vec[vocab_index[u]])
		if y==1 and yp==1:
			color.append(1)
		elif y==1 and yp!=1:
			color.append(2)
		elif y!=1 and yp==1:
			color.append(3)
		else:
			color.append(4)
			
		if y==1 and yp_log==1:
			color_log.append(1)
		elif y==1 and yp_log!=1:
			color_log.append(2)
		elif y!=1 and yp_log==1:
			color_log.append(3)
		else:
			color_log.append(4)
			
		count+=1
		if count==vec_limit:
			break
	return numpy.array(vectors), color, color_log, tag, prec, prec_log

def save_embed_plot(Y,color,tag,prec,clf,fname):
	fig = Plot.figure()
	init = []
	tp = []
	fn = []
	fp = []
	tn = []
	for i,c in enumerate(color):
		if c==0:
			init.append(i)
		elif c==1:
			tp.append(i)
		elif c==2:
			fn.append(i)
		elif c==3:
			fp.append(i)
		else:
			tn.append(i)
	Y_init = Y[init]
	Y_tp = Y[tp]
	Y_fn = Y[fn]
	Y_fp = Y[fp]
	Y_tn = Y[tn]
	Plot.scatter(Y_init[:,0], Y_init[:,1], s=20, c='r', alpha=0.8, label = 'initial adopters', edgecolor='none')
	Plot.scatter(Y_tp[:,0], Y_tp[:,1], s=16, c='g', alpha=0.8, label = 'true positives', edgecolor='none')
	Plot.scatter(Y_fn[:,0], Y_fn[:,1], s=16, c='b', alpha=0.8, label = 'false negatives', edgecolor='none')
	Plot.scatter(Y_fp[:,0], Y_fp[:,1], s=14, c='y', alpha=0.8, label = 'false positives', edgecolor='none')
	Plot.scatter(Y_tn[:,0], Y_tn[:,1], s=10, c='c', alpha=0.4, label = 'true negatives', edgecolor='none')
	Plot.axis('off')
	Plot.legend(prop={'size':8})
	Plot.title('#'+tag+', P@100: '+str(prec)+', '+clf)
	fig.savefig(fname+'.png', dpi=400, bbox_inches='tight')
	
if __name__ == "__main__":
	print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
	for i in range(0,train_ex_limit):
		X,color,color_log,tag,prec,prec_log = get_user_vectors(i)
		if use_tsne==True:
			Y = tsne(X, no_dims = 2, initial_dims = 50, perplexity = 30.0)
		else:
			Y = pca(X, no_dims = 2)
		# with open("/mnt/filer01/word2vec/degree_distribution/adopter_pred_files/single_topic_vis/"+fname+".pickle","wb") as fd:
			# pickle.dump(Y,fd)
		save_embed_plot(Y,color,tag,prec,'RF','embed_adopters_topic_rf'+str(i))
		save_embed_plot(Y,color_log,tag,prec_log,'LR','embed_adopters_topic_lr'+str(i))

#cc 0.0589, candidate set recall 280 out of 4751 cand size 6312
#cc 0.219, candidate set recall 516 out of 2347 cand size 4702
#cc 0.56, candidate set recall 658 out of 1162 cand size 4075