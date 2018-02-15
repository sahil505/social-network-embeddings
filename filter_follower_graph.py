#filter only tweets by subset of users in hashtag sequence file
import time
import sys
import os
import cPickle as pickle
import random

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
		