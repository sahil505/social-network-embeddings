#query users nearest to a given user using node vectors file from distance-filewrite.c file and compare with users in same path in sentences file

import cPickle as pickle
import random
import os, sys, datetime
from heapq import nlargest

start_time = datetime.datetime.now()

adoption_sentence_filename = "/mnt/filer01/word2vec/degree_distribution/sentences_files/userSentencesComb_12hr" #"sample_sequences"
#time_diff_for_edge = 5*1*60*60 #5 context width for path in one direction

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

def call_distance(word):
	return os.system("./distance-filewrite ../node_vectors_12hr.bin query_output_temp12hrsen "+str(word))
	
def get_nearest():
	nearest = []
	with open("query_output_temp12hrsen","rb") as fr:
		for line in fr:
			line=line.rstrip().split('\t')
			nearest.append(int(line[0]))
	return nearest

def compare_nearest(seq,w2v):
	return len(set(seq)&set(w2v))

vocab = []
with open("../node_vocab_12hr.txt","rb") as fr:
	next(fr)
	for line in fr:
		line=line.rstrip().split(' ')
		vocab.append(int(line[0]))
print "Vocab read"

rand_users = random.sample(vocab,100)
rand_users_set = set(rand_users)
vocab = set(vocab)
print "Sample selected"

near_count = [[0]*7697889 for i in xrange(0,100)]

linecount=0
with open(adoption_sentence_filename, 'r') as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		sentence = map(int,u)
		for author in sentence:
			if author in rand_users_set:
				for j in sentence:
					near_count[rand_users.index(author)][j]+=1
				near_count[rand_users.index(author)][author]-=1
		linecount+=1
		if linecount%1000000==0:
			print "path count", linecount
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
	a = call_distance(user)
	if a!=0:
		print "call error"
		sys.exit(0)
	nearest_users_w2v = get_nearest()
	comp_count = compare_nearest(near_users_seq[user][0:100],nearest_users_w2v[0:100])
	print "common users", user, comp_count
	count_pickle.append(comp_count)
	nearest_users_w2v_pickle[user]=nearest_users_w2v

with open("nearest_users_compare12hrsen.pickle","wb") as fd:
	pickle.dump(rand_users,fd)
	pickle.dump(count_pickle,fd)
	pickle.dump(near_users_seq,fd)
	pickle.dump(nearest_users_w2v_pickle,fd)
	pickle.dump(near_count,fd)

print start_time, datetime.datetime.now()