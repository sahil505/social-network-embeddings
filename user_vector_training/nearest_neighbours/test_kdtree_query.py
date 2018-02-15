from scipy.spatial import cKDTree as KDTree
import time
from math import sqrt
import random
from heapq import nsmallest
from sklearn.neighbors import NearestNeighbors

vec_file = "/mnt/filer01/word2vec/node_vectors_1hr_bfsr.txt"
# M=100000
def read_vector_file(path_vectors_file):
	vocab = []
	vectors = []
	count=0
	with open(path_vectors_file,"rb") as fr:
		_,dim = next(fr).rstrip().split(' ')
		word_vector_dim = int(dim)
		next(fr)
		for line in fr:
			# if count==M:
				# break
			line = line.rstrip()
			u = line.split(' ')
			if len(u) != word_vector_dim+1:
				print "vector length error"
			word = int(u[0])
			#normalise to length 1
			# vec = []
			# length = 0.0
			# for d in u[1:]:
				# num=float(d)
				# vec.append(num)
				# length+=num**2
			# length = sqrt(length)
			vec = map(float,u[1:])
			length = sum(x**2 for x in vec)
			vec_norm = [x/length for x in vec]
			vocab.append(word)
			vectors.append(vec_norm)
			count+=1
	return vectors, vocab, word_vector_dim

def get_Nranked_list(query_set_ind,N):
	# wordN = [0]*N
	# distN = [0.0]*N
	dist_total = []
	set_size = len(query_set_ind)
	for i in xrange(0,len(vec)):
		if i in query_set_ind:
			continue
		pres_word = i
		pres_vec = vec[i]
		dist_k = [0.0]*set_size
		k=0
		dim=len(pres_vec)
		for voc_ind in query_set_ind:
			user_vec = vec[voc_ind]
			dist = sum( (user_vec[x]-pres_vec[x])**2 for x in xrange(0,dim) )
			dist_k[k]= sqrt(dist)
			k+=1
		nearest_k = min(dist_k) # dist_k_sorted[0] #  if sorted not needed
		dist_set=nearest_k
		dist_total.append((pres_word,dist_set))
	wordN = [w for w,_ in nsmallest(N,dist_total,key=lambda x: x[1])]
	return wordN #zip(wordN,distN)

t=0.0
t1=0.0
t2=0.0
N=3
k=500
M= 2654594 #1000000
D=10
S=10
eps = 0

vec,vocab,dim = read_vector_file(vec_file)
print "num points", len(vec), "dim", dim

# vec = [v[:D] for v in vec[:M]]
print len(vec),len(vec[0]), "eps", eps
tic = time.clock()
kd = KDTree(vec, leafsize=10)
toc = time.clock()
print "scipy tree built in", (toc-tic)*1000

tic = time.clock()
neigh = NearestNeighbors(n_neighbors=5, radius=1.0, algorithm='ball_tree', leaf_size=10, metric='minkowski', p=2) #'ball_tree', 'kd_tree', 'auto'
kd_sklearn = neigh.fit(vec)
toc = time.clock()
print "sklearn tree built in", (toc-tic)*1000

for _ in range(0,N):	
	sample = random.sample(range(0,M),S)
	sample_vec = [vec[i] for i in sample]
	
	tic = time.clock()
	d_list,knn_list = kd.query(sample_vec,k=k+1) #, eps=eps)
	dist_n_list = []
	for d,n in zip(d_list,knn_list):
		dist_n_list+=list(zip(n,d))[1:]
	knn= [w for w,_ in nsmallest(k,dist_n_list,key=lambda x: x[1])]
	toc = time.clock()
	print "scipy, tree query in", (toc-tic)*1000
	t+=(toc-tic)*1000
	
	
	tic1 = time.clock()
	knn_brute = get_Nranked_list(sample,k)
	toc1 = time.clock()
	print "brute, tree query in", (toc1-tic1)*1000
	if knn_brute!=knn:
		print "scipy, not same points", "same", len(set(knn_brute)&set(knn)), "out of", k
	else:
		print "same", len(set(knn_brute)&set(knn)), len(knn_brute)
	t1+=(toc1-tic1)*1000
	
	tic1 = time.clock()
	d_list,knn_list = neigh.kneighbors(X=sample_vec, n_neighbors=k+1, return_distance=True)
	dist_n_list = []
	for d,n in zip(d_list,knn_list):
		dist_n_list+=list(zip(n,d))[1:]
	knn_sklearn= [w for w,_ in nsmallest(k,dist_n_list,key=lambda x: x[1])]
	toc1 = time.clock()
	print "sklearn, tree query in", (toc1-tic1)*1000
	if knn_sklearn!=knn_brute:
		print "sklearn, not same points", "same", len(set(knn_brute)&set(knn_sklearn)), "out of", k
	else:
		print "same", len(set(knn_brute)&set(knn_sklearn))
	t2+=(toc1-tic1)*1000
	
print "tree query in, avg, kdtree", t*1./N, "brute", t1*1./N, "sklearn", t2*1./N
"""
for i in random.sample(range(0,M),N):
	tic = time.clock()
	_,knn = kd.query(vec[i],k=k) #, eps=eps)
	toc = time.clock()
	# print i, knn
	# print "tree query in", (toc-tic)*1000
	t+=(toc-tic)*1000
	
	tic1 = time.clock()
	knn_brute = get_Nranked_list([i],k)
	toc1 = time.clock()
	# print i, knn_brute
	# print "tree query in", (toc1-tic1)*1000
	if knn_brute!=list(knn):
		print "not same points", "same", len(set(knn_brute)&set(list(knn))), "out of", k
	t1+=(toc1-tic1)*1000
print "tree query in, avg, kdtree", t*1./N, "brute", t1*1./N
"""