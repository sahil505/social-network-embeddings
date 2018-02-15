#get nearest users of the source of a hashtag sequence in test sequences using user vectors and compare with actual adopters in the sequence

import cPickle as pickle
import time
from math import sqrt
import random
from multiprocessing import Pool, cpu_count

NUM_PROCESSES = 5

vec_file = "/mnt/filer01/word2vec/node_vectors_1hr_pr.txt"
adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt" #"sample_sequences"
with open("sequence_file_split_indices.pickle","rb") as fr:
	_ = pickle.load(fr)
	test_seq_id = pickle.load(fr)
test_seq_id = set(test_seq_id)

with open("sequence_file_split_users.pickle","rb") as fr:
	users_train = pickle.load(fr)
	users_test = pickle.load(fr)
users_test = set(users_test)

def read_vector_file(path_vectors_file,users_test):
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
			if word in users_test:
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

vec,vocab,dim = read_vector_file(vec_file,users_test)
vocab_index=dict()
for i in xrange(0,len(vocab)):
	vocab_index[vocab[i]]=i
num_users_test = len(vocab)
# print "num users in test sequences", num_users_test
# print "users removed from vocab", len(set(users_train)-set(vocab))
# print "users in test sequences but not in vocab", len(users_test-set(vocab))

#Peter Norvig's code for memo
# def memo(f):
#     "Memoize function f."
#     table = {}
#     def fmemo(*args):
#         if args not in table:
#             table[args] = f(*args)
#         return table[args]
#     fmemo.memo = table
#     return fmemo
# dist_memo = dict()

# @memo
def get_Nranked_list(query,N):
	wordN = [0]*N
	distN = [0.0]*N
	try:
		voc_ind = vocab_index[query]
	except KeyError:
		print "query word not present"
		return
	query_vec = vec[voc_ind]
	for i in range(0,len(vec)):
		if i==voc_ind:
			continue
		pres_word = vocab[i]
		pres_vec = vec[i]
		dist = 0.0
		for x in range(0,dim):
			dist+=query_vec[x]*pres_vec[x]
		#dist = sum(query_vec[x]*pres_vec[x] for x in range(0,dim))
		for j in range(0,N):
			if dist>distN[j]:
				for k in range(N-1,j,-1):
					distN[k] = distN[k-1]
					wordN[k] = wordN[k-1]
				distN[j] = dist
				wordN[j] = pres_word
				break
	return wordN #zip(wordN,distN)

not_found_vocab=[]
# source_thr = 1395858601 + 7*24*60*60
tag_seq = []
count=0
# nb_seq = dict()
with open(adoption_sequence_filename, "rb") as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		not_found=0
		# first_timestamp = int(u[1][0:u[1].index(',')])
		# if first_timestamp>=source_thr
		if count in test_seq_id:
			seq=[]
			for i in xrange(1, len(u)):
				#timestamp = int(u[i][0:u[i].index(',')])
				author = int(u[i][u[i].index(',')+1 : ])
				if author in vocab_index:
					seq.append(author)
				else:
					not_found+=1
			if len(seq)>1:
				tag_seq.append(seq)
				not_found_vocab.append(not_found)
		# else:
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
# nb_seq_part = [(a,nb_seq[a]) for a in nb_seq]
# nb_seq_part_sorted = sorted(nb_seq_part, key=lambda x: x[1], reverse=True)
# nb_seq_order = [a for a,_ in nb_seq_part_sorted]
# pickle.dump(nb_seq_order,open("adopter_pred_files/baseline_user_order_bfsr.pickle","wb"))
nb_seq_order = pickle.load(open("adopter_pred_files/baseline_user_order_pr.pickle","rb"))
print len(nb_seq_order)
# print len(tag_seq),len(test_seq_id),count
# print sum(not_found_vocab)/float(len(not_found_vocab)),max(not_found_vocab),min(not_found_vocab)

seq_random_index=range(0,len(tag_seq))
random.shuffle(seq_random_index)

def adopter_prediction(process_num,start,end):
	seq_count_limit=100
	num_seqs=0
	mean_ap=0
	# mean_prec_r=0
	mean_ap_nbapp=0
	# mean_prec_r_nbapp=0
	# N=100
	for i in seq_random_index[start:end]:
		seq_sample_vocab = tag_seq[i]
		# source_user=seq_sample[0]
		# if source_user not in vocab_index:
		# 	continue
		# seq_sample_vocab = [x for x in seq_sample if x in vocab_index]
		# if len(seq_sample_vocab)<2:#2
		# 	continue
		source_user=seq_sample_vocab[0]
		seq_sample_vocab = set(seq_sample_vocab[1:])
		M = len(seq_sample_vocab)
		N = num_users_test #M #1000
		# if M<1000:
		# 	continue
		not_found=not_found_vocab[i]
		#source_vec=vec[vocab_index[source_user]]

		adopters_vec = get_Nranked_list(source_user,N)
		precision_k = 0.0
		num_hits = 0.0
		for k,p in enumerate(adopters_vec):
			if p in seq_sample_vocab:
				num_hits+=1.0
				precision_k += num_hits/(k+1.0)
		average_precision = precision_k/min(M,N)
		# prec_r = num_hits/M
		print "Avg precision", average_precision, "num of users not found", not_found, "num of adopters in seq", len(seq_sample_vocab), "Process", process_num
		# print "Precision", num_hits/N, "Recall", num_hits/M
		mean_ap+=average_precision
		# mean_prec_r+=prec_r
		num_seqs+=1
		print "MAP", mean_ap/float(num_seqs), "Process", process_num#, "MRP", mean_prec_r/float(num_seqs)
		
		nb_seq_order = nb_seq_order[:N]
		precision_k_nbapp = 0.0
		num_hits_nbapp = 0.0
		for k,p in enumerate(nb_seq_order):
			if p in seq_sample_vocab:
				num_hits_nbapp+=1.0
				precision_k_nbapp += num_hits_nbapp/(k+1.0)
		average_precision_nbapp = precision_k_nbapp/min(M,N)
		# prec_r_nbapp = num_hits_nbapp/M
		print "Nb_App", "Avg precision", average_precision_nbapp, "Process", process_num
		# print "Precision", num_hits_nbapp/N, "Recall", num_hits_nbapp/M
		mean_ap_nbapp+=average_precision_nbapp
		# mean_prec_r_nbapp+=prec_r_nbapp
		print "Nb_App", "MAP", mean_ap_nbapp/float(num_seqs), "Process", process_num#, "MRP", mean_prec_r_nbapp/float(num_seqs)
		
		seq_count_limit-=1
		if seq_count_limit==0:
			break
	print num_seqs, mean_ap, mean_ap_nbapp, "Process", process_num
	print "user vectors", mean_ap/float(num_seqs), "Process", process_num
	print "Nb_App", mean_ap_nbapp/float(num_seqs), "Process", process_num
	# print mean_prec_r/float(num_seqs)
	#pickle.dump(source_time,open("source_time.pickle","wb"))

num_workers = min(NUM_PROCESSES,cpu_count())
pool = Pool(processes=num_workers) 
process_num=0
NUM_SEQ = len(seq_random_index)
lines_per_process = int(NUM_SEQ/(2.0*num_workers))
for s,e in ( (i,min(i+lines_per_process,NUM_SEQ)) for i in xrange(0,NUM_SEQ,lines_per_process) ):
	pool.apply_async(adopter_prediction, args=(process_num,s,e))
	process_num+=1
pool.close()
pool.join()