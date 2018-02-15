#TODO
#distribution of number of tweets sharing the same hashtags as the ones used by each user, number of users with atleast 20% common following users and number of reciprocal relations with users in the subset of selected users
#use these features as sentences in word2vec for node representations or try deepwalk on adjacency list of subset of users or use sequence of authors adopting a hashtag for hashtags with atleast 10 adoptions as sentences
import time
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
"""
tags_for_user = dict()
num_tweets_per_tag = dict()
with open('/twitterSimulations/timeline_data/dif_timeline1s', 'r') as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split('\t')
		tag = u[0]
		author = m[int(u[2])]
		if author not in tags_for_user:
			tags_for_user[author]=set()
		tags_for_user[author].add(tag)
		if tag not in num_tweets_per_tag:
			num_tweets_per_tag[tag]=0
		num_tweets_per_tag[tag]+=1
print len(tags_for_user)
"""
selected_users = set()
with open("userSubset.csv","r") as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(',')
		id,_,_ = int(u[0]),int(u[1]),int(u[2])
		selected_users.add(id)
		
#subset follower and friend adjacency list
"""
arr = ["user_followers_bigger_graph.txt","user_followers_bigger_graph_2.txt", "user_followers_bigger_graph_i.txt","user_followers_bigger_graph_recrawl_2.txt", "user_followers_bigger_graph_recrawl_3.txt","user_followers_bigger_graph_recrawl.txt"]
follower = dict()
for i in arr:
	fr = open("/twitterSimulations/graph/" + i,'r')
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		if(int(u[0]) > 7697889):
			continue
		node = m[int(u[1])]
		if node not in selected_users:
			continue
		follower[node] = []
		if len(u) > 2:
			for j in range(2,len(u)):
				snode = m[int(u[j])]
				if snode in selected_users:
					follower[node].append(snode)
	fr.close()
	print i
pickle.dump( follower, open( "subset_follower_graph.pickle", "wb" ) )

arr_friend = ["user_friends_bigger_graph.txt","user_friends_bigger_graph_2.txt", "user_friends_bigger_graph_i.txt","user_friends_bigger_graph_recrawl.txt"]
friend = dict()
num_friend_id_not_found=0
friend_id_not_found=set()
for i in arr_friend:
	fr = open("/twitterSimulations/graph/" + i,'r')
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		if(int(u[0]) > 7697889):
			continue
		try:
			node = m[int(u[1])]
		except:
			num_friend_id_not_found += 1
			friend_id_not_found.add(int(u[1]))
			continue
		if node not in selected_users:
			continue
		friend[node] = []
		if len(u) > 2:
			for j in range(2,len(u)):
				try:
					snode = m[int(u[j])]
				except:
					num_friend_id_not_found += 1
					friend_id_not_found.add(int(u[1]))
					continue
				if snode in selected_users:
					friend[node].append(snode)
	fr.close()
	print i
pickle.dump( friend, open( "subset_friend_graph.pickle", "wb" ) )
print num_friend_id_not_found
pickle.dump( friend_id_not_found, open( "friend_id_not_found.pickle", "wb" ) )
"""

follower = pickle.load( open( "subset_follower_graph.pickle", "rb" ) )
print "Follower file loaded"
"""
friend = pickle.load( open( "subset_friend_graph.pickle", "rb" ) )
print "Friend file loaded"
"""
#number of users with reciprocal links
"""
num_rec = dict()
count=0
for node in selected_users:
	count+=1
	if count%10000==0:
		print count," Users processed"
	num_rec[node]=0
	# for nbh in friend[node]:
		# if node in friend[nbh]:
			# num_rec[node]+=1
	try:
		incoming = set(friend[node])
		outgoing = set(follower[node])
		reciprocal = set.intersection(incoming, outgoing)
		num_rec[node]+=len(reciprocal)
	except:
		pass
pickle.dump( num_rec, open( "num_reciprocal_links.pickle", "wb" ) )
"""
"""
num_rec = pickle.load( open( "num_reciprocal_links.pickle", "rb" ) )
"""
#users with more than 20% common friends
num_common_friends = dict()
count=0
for node in selected_users:
	count+=1
	if count%1000==0:
		print count," Users processed"
	try:
		out_nodes = follower[node]
	except:
		continue
	num_out = len(out_nodes)#len(out_nodes)
	for i in range(0,num_out):
		out = out_nodes[i]
		for j in range(i+1,num_out):
			sout = out_nodes[j]
			if out>sout:
				(out,sout) = (sout,out)
			if out not in num_common_friends:
				num_common_friends[out]=dict()
				num_common_friends[out][sout]=1
			elif sout not in num_common_friends[out]:
				num_common_friends[out][sout]=1
			else:
				num_common_friends[out][sout]+=1
		
pickle.dump( num_common_friends, open( "num_common_friends.pickle", "wb" ) )
"""
num_common_friends_thr = dict()
for out in num_common_friends:
	for sout in num_common_friends[out]:
		thr = .20*len(friend[out])
		thr_s = .20*len(friend[sout])
		common = num_common_friends[out][sout]
		if common>=thr:
			if out not in num_common_friends_thr:
				num_common_friends_thr[out]=0
			num_common_friends_thr[out]+=1
		if common>=thr_s:
			if sout not in num_common_friends_thr:
				num_common_friends_thr[sout]=0
			num_common_friends_thr[sout]+=1

pickle.dump( num_common_friends_thr, open( "num_common_friends_thr.pickle", "wb" ) )
"""
"""
#tweets with same hashtags
num_tweets_with_same_tags = dict()
for node in selected_users:
	num_tweets_with_same_tags[node] = sum([num_tweets_per_tag[x] for x in tags_for_user[node]])
	
with open("featuresUserSubset.csv","w") as fd:
	for i in selected_users:
		fd.write(str(i)+","+str(num_common_friends_thr[i])+","+str(num_rec[i])+","+str(num_tweets_with_same_tags[i])+"\n")
"""