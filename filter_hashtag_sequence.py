#filter only tweets by subset of users in hashtag sequence file
import time
import sys
import os
import cPickle as pickle
import random

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

with open('hashtagAdoptionSequences.txt','wb') as fd: # 'hashtagAdoptionSequences_filter.txt'
	for tag in adoption_sequence.keys():
		if len(adoption_sequence[tag])>=min_tweets_sequence:
			fd.write(tag)
			for t,a in adoption_sequence[tag]:
				fd.write(' '+str(t)+','+str(a)) #author is of type str for using join
			fd.write('\n')