#same as distance.c file in word2vec for use in query_nearest_users.py

from math import sqrt

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

def get_Nnearest(query,vec,vocab,N):
	wordN = [0]*N
	distN = [0.0]*N
	try:
		voc_ind = vocab.index(query)
	except ValueError:
		print "query word not present"
		return
	query_vec = vec[voc_ind]
	dim = len(query_vec)
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

def get_distance(query1,query2,vec,vocab):
	dist=0.0
	try:
		vec1=vec[vocab.index(query1)]
		vec2=vec[vocab.index(query2)]
	except ValueError:
		print "query word not present"
		return
	for i in range(0,len(vec1)):
		dist+=vec1[i]*vec2[i]
	return dist

#vec,vocab,_ = read_vector_file("/mnt/filer01/word2vec/node_vectors_1hr.txt")
#print get_Nnearest(17,vec,vocab,N=1)
#print get_distance(17,1145375,vec,vocab)
#print get_distance(1,1145375,vec,vocab)==None