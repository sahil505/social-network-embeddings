#filter only tweets by subset of users in hashtag sequence file
import time
import sys
import os
import cPickle as pickle
import random
"""	
min_tweets_sequence = 2 
selected_users = set()
with open("userSubset.csv","r") as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(',')
		id,_,_ = int(u[0]),int(u[1]),int(u[2])
		selected_users.add(id)

m = dict()
fr = open("/twitterSimulations/graph/map.txt")
for line in fr:
	line = line.rstrip()
	u = line.split(' ')
	m[int(u[0])] = int(u[1])
fr.close()
print 'Map Read'

adoption_sequence = dict()
with open('/twitterSimulations/timeline_data/dif_timeline1s', 'r') as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split('\t')
		tag = u[0]
		time = int(u[1])
		author = m[int(u[2])]
		if author not in selected_users:
			continue
		try:
			adoption_sequence[tag].append((time,author))
		except KeyError:
			adoption_sequence[tag]=[(time,author)]
print len(adoption_sequence)

with open('hashtagAdoptionSequences_filter.txt','wb') as fd:
	for tag in adoption_sequence.keys():
		if len(adoption_sequence[tag])>=min_tweets_sequence:
			fd.write(tag)
			for t,a in adoption_sequence[tag]:
				fd.write(' '+str(t)+','+str(a)) #author is of type str for using join
			fd.write('\n')
"""
#separate sequences into training (80%) and test sequences (20%)
"""
adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt"
adoption_sequence = []
large_tag_id = []
count=0
with open(adoption_sequence_filename, 'r') as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		#tag = u[0]
		sequence = []
		if len(u)-1>=100000:
			large_tag_id.append(count)
		for i in range(1, len(u)):
			#timestamp = int(u[i][0:u[i].index(',')])
			author = int(u[i][u[i].index(',')+1 : ])
			sequence.append(author)
		adoption_sequence.append(sequence)
		count+=1

num_lines = len(adoption_sequence) #3617312
print num_lines
seq_random_index=range(0,num_lines)
random.shuffle(seq_random_index)
num_train = int(0.8*num_lines)
print num_train
train_seq_id = seq_random_index[:num_train]
test_seq_id = seq_random_index[num_train:]
with open("sequence_file_split_indices.pickle","wb") as fd:
	pickle.dump(train_seq_id,fd)
	pickle.dump(test_seq_id,fd)
users_train=set()
for i in train_seq_id:
	for u in adoption_sequence[i]:
		users_train.add(u)
users_test=set()
overlap = set()
for i in test_seq_id:
	for u in adoption_sequence[i]:
		users_test.add(u)
		if u in users_train:
			overlap.add(u)
print len(users_train), len(users_test), len(overlap)
with open("sequence_file_split_users.pickle","wb") as fd:
	pickle.dump(users_train,fd)
	pickle.dump(users_test,fd)

# with open("sequence_large_hashtags.pickle","wb") as fd:
# 	pickle.dump(large_tag_id,fd)
"""
# filter follower files for users in adoption sequence
adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt"
adoption_sequence_users = set()
count=0
with open(adoption_sequence_filename, 'r') as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		#tag = u[0]
		for i in range(1, len(u)):
			#timestamp = int(u[i][0:u[i].index(',')])
			author = int(u[i][u[i].index(',')+1 : ])
			adoption_sequence_users.add(author)
		count+=1
print len(adoption_sequence_users), count

m = dict()
fr = open("/twitterSimulations/graph/map.txt")
for line in fr:
	line = line.rstrip()
	u = line.split(' ')
	m[int(u[0])] = int(u[1])
fr.close()
print 'Map Read'

# arr = ["user_followers_bigger_graph_recrawl_3.txt"]
arr = ["user_followers_bigger_graph.txt","user_followers_bigger_graph_2.txt", "user_followers_bigger_graph_i.txt","user_followers_bigger_graph_recrawl_2.txt", "user_followers_bigger_graph_recrawl_3.txt","user_followers_bigger_graph_recrawl.txt"]

follower_adj = [ [] for i in xrange(0, 7697889) ]

for i in arr:
	fr = open("/twitterSimulations/graph/" + i,'r')
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		if(int(u[0]) > 7697889):
			continue
		if len(u) > 2:
			for j in range(2,len(u)):
				follower_adj[m[int(u[1])]].append(m[int(u[j])])
	fr.close()
	print i

print 'Graph Read\n'

# for i in range(0, 7697889):
	# follower_adj[i] = set(follower_adj[i])

print 'Graph Set\n'

with open("graph_files/follower_graph_tweeters","wb") as fd:
	for i in follower_adj:
		if i in adoption_sequence_users:
			fol = set(follower_adj[i])&adoption_sequence_users
			fol = map(str,list(fol))
			fd.write(str(len(fol))+" "+str(i)+" "+" ".join(fol)+"\n")
		