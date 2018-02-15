#filter only tweets by subset of users in hashtag sequence file
import time
import sys
import os
import cPickle as pickle
import random

#separate sequences into training (80%) and test sequences (20%)
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

with open("sequence_large_hashtags.pickle","wb") as fd:
	pickle.dump(large_tag_id,fd)