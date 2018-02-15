#plot scatterplot of similarity between nearest users for a query users obtained from user vectors and from hashtag sequence file

import cPickle as pickle
from distance_w2v import *
import matplotlib
matplotlib.use('Agg')
import pylab as Plot
from numpy import array

vec_file = "/mnt/filer01/word2vec/node_vectors_1hr.txt"
nearest_users_pickle = "/mnt/filer01/word2vec/degree_distribution/nearest_users_compare1hr_5.pickle"

def save_scatterplot(X,overlap,fname):
	posX,posY,color = zip(*X)
	#max_d = max(color)
	#min_d = min(color)
	#color_norm = [(x-min_d)/float(max_d-min_d) for x in color]
	fig = Plot.figure()
	Plot.scatter(posX, posY, s=20, c=color)
	#Plot.axis('off')
	Plot.xlim([0,100])
	Plot.ylim([0,100])
	Plot.xlabel('User vectors')
	Plot.ylabel('Counts')
	Plot.colorbar()
	fig.suptitle('Overlap '+str(overlap))
	fig.savefig(fname, dpi=100, bbox_inches='tight')

with open(nearest_users_pickle,"rb") as fr:
	sample_users = pickle.load(fr)
	overlap_count = pickle.load(fr)
	nearest_users_seq = pickle.load(fr)
	nearest_users_w2v = pickle.load(fr)

max_overlap = overlap_count.index(max(overlap_count))
min_overlap = overlap_count.index(min(overlap_count))
overlap_query_users = [max(overlap_count),min(overlap_count)]

vec,vocab,_ = read_vector_file(vec_file)

query_users = [sample_users[max_overlap],sample_users[min_overlap]]
count=0
for query_user in query_users:
	count+=1
	users_seq = nearest_users_seq[query_user]
	users_w2v = nearest_users_w2v[query_user]
	print len(users_seq),len(users_w2v)
	X = []
	for i in range(0,len(users_w2v)):
		vec1=vec[vocab.index(users_w2v[i])]
		for j in range(0,len(users_seq)):
			vec2=vec[vocab.index(users_seq[j])]
			dist = 0.0
			for d in range(0,len(vec1)):
				dist+=vec1[d]*vec2[d]
			X.append((i+1,j+1,dist))
	save_scatterplot(X,overlap_query_users[count-1],fname='nearest_users_scatterplot'+str(count))