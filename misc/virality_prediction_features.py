#write features for hashtags in virality prediction using user vectors

import cPickle as pickle
import time
from distance_w2v import *

vec_file = "/mnt/filer01/word2vec/node_vectors_1hr.txt"
timeline_file = "/twitterSimulations/timeline_data/timeline_weng"
feature_file = "/mnt/filer01/word2vec/degree_distribution/feature_file.csv"

vec,vocab,dim = read_vector_file(vec_file)
vocab_index=dict()
for i in range(0,len(vocab)):
	vocab_index[vocab[i]]=i

m = dict()
fr = open("/twitterSimulations/graph/map.txt")
for line in fr:
	line = line.rstrip()
	u = line.split(' ')
	m[int(u[0])] = int(u[1])
fr.close()
print 'Map Read'

not_found_vocab=[]
pred_thr = 1500
with open(timeline_file, "rb") as fr, open(feature_file, "wb") as fd:
	feature_names = ','.join(["max_"+str(x) for x in range(0,dim)])+','+','.join(["min_"+str(x) for x in range(0,dim)])+','+','.join(["avg_"+str(x) for x in range(0,dim)])
	fd.write("TagName,"+feature_names+",Class\n")
	for line in fr:	
		line = line.rstrip()
		u = line.split(' ')
		if len(u) <= pred_thr:
			continue
		numTweets = 0
		not_found=0
		user_vectors = []
		for i in range(1, len(u)):
			#timestamp = int(u[i][0:u[i].index(',')])
			numTweets = i
			if(numTweets > pred_thr):
				break
			author = int(u[i][u[i].index(',')+1 : ])
			author = m[author]
			if author in vocab_index:
				user_vec=vec[vocab_index[author]]
			else:
				not_found+=1
				continue
			user_vectors.append(user_vec)
		if user_vectors==[]:
			max_vec = [0.0]*dim
			min_vec = [0.0]*dim
			avg_vec = [0.0]*dim
			print u[0]
		else:
			aggr_vec = zip(*user_vectors)
			max_vec = []
			min_vec = []
			avg_vec = []
			for i in range(0,len(aggr_vec)):
				d = aggr_vec[i]
				max_vec.append(max(d))
				min_vec.append(min(d))
				avg_vec.append(sum(d)/float(len(d)))
		if len(u) > 10000:
			class_label = '1'
		else:
			class_label = '0'
		fd.write(str(u[0])+','+','.join(map(str,max_vec))+','+','.join(map(str,min_vec))+','+','.join(map(str,avg_vec))+','+class_label+'\n')
		not_found_vocab.append(not_found)

print sum(not_found_vocab)/float(len(not_found_vocab)),max(not_found_vocab),min(not_found_vocab)
#pickle.dump(not_found_vocab,open("not_found_vocab.pickle","wb"))