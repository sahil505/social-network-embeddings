#query users nearest to a given user using node vectors file from distance-filewrite.c file and compare with users nearby in hashtag sequence file

import cPickle as pickle
import random
import os, sys, datetime
from heapq import nlargest
from distance_w2v import *

start_time = datetime.datetime.now()

adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt" #"sample_sequences"
time_diff_for_edge = 1*1*60*60 #5 context width for path in one direction
vec_file = "../node_vectors_1hr_bfs_sgng.txt"
vocab_file = "../node_vocab_1hr_bfs_sgng.txt"
out_file = "nearest_users_compare1hr_bfs_sgng.pickle"
vec,vocab_ind,_ = read_vector_file(vec_file)

m = dict()
fr = open("/twitterSimulations/graph/map.txt")
for line in fr:
	line = line.rstrip()
	u = line.split(' ')
	m[int(u[0])] = int(u[1])
fr.close()
print 'Map Read'

location_buckets = [-1] * 7697889
fr = open('/twitterSimulations/known_locations.txt', 'r')
for line in fr:
	line = line.rstrip()
	u = line.split('\t')
	try:
		location_buckets[m[int(u[0])]] = int(u[1])
	except:
		pass
fr.close()

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

# def call_distance(word):
# 	return os.system("./distance-filewrite ../node_vectors_1hr_bfs_15.bin query_output_temp1hr_bfs_15 "+str(word))
	
# def get_nearest():
# 	nearest = []
# 	with open("query_output_temp1hr_bfs_15","rb") as fr:
# 		for line in fr:
# 			line=line.rstrip().split('\t')
# 			nearest.append(int(line[0]))
# 	return nearest

def compare_nearest(seq,w2v):
	return len(set(seq)&set(w2v))

vocab = []
freq = dict()
with open(vocab_file,"rb") as fr:
	next(fr)
	for line in fr:
		line=line.rstrip().split(' ')
		vocab.append(int(line[0]))
		freq[int(line[0])]=int(line[1])
print "Vocab read"

sub_vocab=[]
for v in vocab:
	if freq[v]>10000:
		sub_vocab.append(v)
rand_users = random.sample(vocab,100)
rand_users_set = set(rand_users)
vocab = set(vocab)
print "Sample selected"

near_count = [[0]*7697889 for i in xrange(0,100)]

tagcount=0
with open(adoption_sequence_filename, 'r') as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		for i in range(1, len(u)):
			timestamp = int(u[i][0:u[i].index(',')])
			author = int(u[i][u[i].index(',')+1 : ])
			location_user = location_buckets[author]
			if author in rand_users_set:
				for j in range(i+1, len(u)):
					t1 = int(u[j][0:u[j].index(',')])
					a1 = int(u[j][u[j].index(',')+1 : ])
					if t1-timestamp<=time_diff_for_edge:
						if location_buckets[a1]==location_user:
							near_count[rand_users.index(author)][a1]+=1
					else:
						break
				for j in range(i-1, 0, -1):
					t1 = int(u[j][0:u[j].index(',')])
					a1 = int(u[j][u[j].index(',')+1 : ])
					if timestamp-t1<=time_diff_for_edge:
						if location_buckets[a1]==location_user:
							near_count[rand_users.index(author)][a1]+=1
					else:
						break
		tagcount+=1
		if tagcount%100000==0:
			print "Hashtag count", tagcount
print "Sequence file read"

near_users_seq = dict()
for i in range(0,len(rand_users)):
	user_count = near_count[i]
	count = []
	for l in vocab:#xrange(0,7697889):
		if user_count[l]!=0 and l!=rand_users[i]:
			count.append((l,user_count[l]))
	#count = zip(range(0,7697889),near_count[i])
	#count_nz = [(a,b) for (a,b) in count if b!=0]
	#count_s = sorted(count_nz,key=lambda x: x[1],reverse=True)[0:100]
	#count_s = sorted(range(0,7697889),key=lambda x: user_count[x],reverse=True)
	count_s = nlargest(100,count,key=lambda x: x[1])
	if len(count_s)==0:
		u,c = [], []
	else:
		u,c = zip(*count_s)
	near_users_seq[rand_users[i]]=list(u)
	print "sel count", rand_users[i], len(u), "non zero", len(count)

nearest_users_w2v_pickle = dict()
count_pickle = []
for user in rand_users:
	# a = call_distance(user)
	# if a!=0:
	# 	print "call error"
	# 	sys.exit(0)
	# nearest_users_w2v = get_nearest()
	nearest_users_w2v = get_Nnearest(user,vec,vocab_ind,100)
	comp_count = compare_nearest(near_users_seq[user][0:100],nearest_users_w2v[0:100])
	print "common users", user, comp_count, "out of", min(len(near_users_seq[user]),len(nearest_users_w2v))
	count_pickle.append(comp_count)
	nearest_users_w2v_pickle[user]=nearest_users_w2v

with open(out_file,"wb") as fd:
	pickle.dump(rand_users,fd)
	pickle.dump(count_pickle,fd)
	pickle.dump(near_users_seq,fd)
	pickle.dump(nearest_users_w2v_pickle,fd)

print start_time, datetime.datetime.now()