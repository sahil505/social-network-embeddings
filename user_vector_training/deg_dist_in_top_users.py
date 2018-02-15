#filter users according to number of tweets with hashtags and number of following, plot degree distribution of this subset of users to check if there is senough context available for each user
import time
import re
import datetime
import dateutil.tz
import calendar
import sys
import os
import cPickle as pickle
	

m = dict()
fr = open("/twitterSimulations/graph/map.txt")
for line in fr:
	line = line.rstrip()
	u = line.split(' ')
	m[int(u[0])] = int(u[1])
fr.close()

num_tagtweets_per_user = dict()
with open('/twitterSimulations/timeline_data/dif_timeline1s', 'r') as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split('\t')
		author = m[int(u[2])]
		if author not in num_tagtweets_per_user:
			num_tagtweets_per_user[author]=0
		num_tagtweets_per_user[author]+=1
print len(num_tagtweets_per_user)
selected_tagtweets_users = set(num_tagtweets_per_user.keys())
"""
with open("numTweetsPerAuthor.csv","w") as fd:
	for i in num_tagtweets_per_user:
		fd.write(str(i)+","+str(num_tagtweets_per_user[i])+"\n")
"""
node_nbh = pickle.load(open( "/twitterSimulations/friends_count_user.pickle", "rb" ) )
print len(node_nbh)

selected_friends_users = set(node_nbh.keys())
"""
with open("numFriendsPerUser.csv","w") as fd:
	for i in node_nbh:
		fd.write(str(i)+","+str(node_nbh[i])+"\n")
"""
common_users = set.intersection(selected_tagtweets_users, selected_friends_users)
print len(common_users)

def get_subset(d,t):
	s = set()
	for i in d:
		if d[i]>=t:
			s.add(i)
	return s
sel_tagtweets = get_subset(num_tagtweets_per_user,15)
sel_friends = get_subset(node_nbh,200)

common_users = set.intersection(sel_tagtweets, sel_friends) #1001525
print len(common_users)

with open("userSubset.csv","w") as fd:
	for i in common_users:
		fd.write(str(i)+","+str(num_tagtweets_per_user[i])+","+str(node_nbh[i])+"\n")