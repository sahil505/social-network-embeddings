from heapq import nsmallest,nlargest
from math import sqrt
vec = [(1,1),(4,2),(2,2),(3,2),(3,3),(4,4),(2,3)]
for i in range(0,7):
    a,b=vec[i]
    l=float(sqrt(a**2+b**2))
    vec[i]=(a/l,b/l)
vocab = [1,2,3,4,5,6,7]
dim = 2
par_m = 2
vocab_index=dict()
for i in xrange(0,len(vocab)):
	vocab_index[vocab[i]]=i
query_set = [3,4]
N=4

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
	print query_set_ind
	for i in xrange(0,len(vec)):
		if i in query_set_ind:
			continue
		pres_word = vocab[i]
		pres_vec = vec[i]
		dist_k = [0.0]*set_size
		k=0
		for voc_ind in query_set_ind:
			user_vec = vec[voc_ind]
			#Euclidean distance, cosine similarity user_vec[x]*pres_vec[x], change to decreasing order of distance in sorted,distN
			print user_vec,pres_vec
			dist = 1- sum((user_vec[x]*pres_vec[x]) for x in xrange(0,dim))
			dist_k[k]=sqrt(float(2*dist))
			k+=1
			# dist = 0.0
			# for x in xrange(0,dim):
			# 	dist+=(user_vec[x]-pres_vec[x])**2 
		#distance of a point from a set
		# dist_k_sorted = sorted(dist_k)
		print i,dist_k
		nearest_k = min(dist_k) # dist_k_sorted[0] #  if sorted not needed
		if nearest_k!=0.0:
			dist_set=sum( (nearest_k/dist_k[p])**(par_m) for p in xrange(0,set_size) )
			dist_set = nearest_k * (dist_set)**(1.0/set_size)
		else:
			dist_set=0.0
		print i,dist_set
		dist_total.append((pres_word,dist_set))
		# for j in xrange(0,N):
		# 	if dist>distN[j]:
		# 		for k in xrange(N-1,j,-1):
		# 			distN[k] = distN[k-1]
		# 			wordN[k] = wordN[k-1]
		# 		distN[j] = dist
		# 		wordN[j] = pres_word
		# 		break
	print dist_total
	wordN = [w for w,_ in nsmallest(N,dist_total,key=lambda x: x[1])]
	return wordN #zip(wordN,distN)

print get_Nranked_list(query_set,N)

adj = {1:set([2,3]),2:set([1,3,5,6]),3:set([1]),4:set([5]),5:set([1,2]),6:set([1])}
nb_seq_order = [3,4,5,1,2,6,7]
def getadj(user):
	return adj[user]
def get_Nranked_list_fol(query_set,N):
	friend_count = dict()
	init_adopters = query_set
	sec_hop = 2
	while (sec_hop>0):
		for a in init_adopters:
			followers = getadj(a)
			print a,followers
			for f in followers-set(query_set):
				try:
					friend_count[f]+=1
				except KeyError:
					friend_count[f]=1
		init_adopters = friend_count.keys()
		sec_hop-=1
		print friend_count
	friend_count_list = [(f,friend_count[f]) for f in friend_count]
	print friend_count_list
	ranked_list = [f for f,_ in nlargest(N,friend_count_list,key=lambda x: x[1])]
	print ranked_list
	if len(friend_count_list)>=N:
		return ranked_list
	else:
		print "followers ranked list short"
		users_left = N-len(friend_count_list)
		for i in nb_seq_order:
			if i not in friend_count and i not in query_set:
				ranked_list.append(i)
				users_left-=1
			if users_left==0:
				break
		return ranked_list

print get_Nranked_list_fol(query_set,N)

num_init_adopters=2
N = 3
seq_sample_vocab = [3,4,1,7,2]
init_adopters=seq_sample_vocab[0:num_init_adopters]
seq_sample_vocab = set(seq_sample_vocab[num_init_adopters:])
M = len(seq_sample_vocab)
print M, "pred seq length"
#precision, recall evaluation
adopters_vec = get_Nranked_list(init_adopters,N)
print adopters_vec
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
print "Avg precision", average_precision, "adopters in seq", len(seq_sample_vocab)
# print "RPrecision", prec_r
print "Precision", prec_k, "Recall", rec_k

adoption_sequence_filename="ab.txt"
seq_len_threshold=3
def read_adoption_sequence(adoption_sequence_filename, start, end,train_seq_id,large_tag_id):
	with open(adoption_sequence_filename, 'r') as fr:
		count=0
		for line in fr:
			if count < start:
				count+=1
				continue
			elif count >= end:
				return
			if count not in train_seq_id or count in large_tag_id:
				count+=1
				continue
			count+=1
			line = line.rstrip()
			u = line.split(' ')
			tag = u[0]
			sequence = []
			adopters = set()
			for i in range(1, len(u)):
				timestamp = int(u[i][0:u[i].index(',')])
				author = int(u[i][u[i].index(',')+1 : ])
				sequence.append((timestamp,author))
				adopters.add(author)
			if len(adopters) < seq_len_threshold:
				continue
			yield (tag,sequence)
for i in read_adoption_sequence(adoption_sequence_filename, 0, 4,set([0,1,3]),[]):
	print i