#TODO
#distribution of number of tweets sharing the same hashtags as the ones used by each user, number of users with atleast 20% common following users and number of reciprocal relations with users in the subset of selected users
#use these features as sentences in word2vec for node representations or try deepwalk on adjacency list of subset of users or use sequence of authors adopting a hashtag for hashtags with atleast 10 adoptions as sentences
import time
import sys
import os
import cPickle as pickle
import random


m = dict()
fr = open("/twitterSimulations/graph/map.txt")
for line in fr:
	line = line.rstrip()
	u = line.split(' ')
	m[int(u[0])] = int(u[1])
fr.close()

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
"""
follower = pickle.load( open( "subset_follower_graph.pickle", "rb" ) )
print "Follower file loaded"
"""
friend = pickle.load( open( "subset_friend_graph.pickle", "rb" ) )
print "Friend file loaded"

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

num_rec = pickle.load( open( "num_reciprocal_links.pickle", "rb" ) )


def get_intersection(list1,list2):
	s = set(list2)
	count=0
	for i in list1:
		if i in s:
			count+=1
	return count
	
#users with more than 20% common friends
num_common_friends = dict()
count=0
num_common_friends_thr = dict()
selected_users_list = random.sample(selected_users,500)
for i in range(0,len(selected_users_list)):
	count+=1
	if count%1000==0:
		print count," Users processed"
	node = selected_users_list[i]
	adj_nodes = friend[node]
	thr = .20*len(adj_nodes)
	
	# for j in range(i+1,len(selected_users_list)):
		# snode = selected_users_list[j]
	for snode in selected_users:
		
		nbh_adj_nodes = friend[snode]
		thr_s = .20*len(nbh_adj_nodes)
		common = get_intersection(adj_nodes, nbh_adj_nodes)
		if common>=thr:
			if node not in num_common_friends_thr:
				num_common_friends_thr[node]=0
			num_common_friends_thr[node]+=1
		# if common>=thr_s:
			# if snode not in num_common_friends_thr:
				# num_common_friends_thr[snode]=0
			# num_common_friends_thr[snode]+=1
pickle.dump( num_common_friends_thr, open( "num_common_friends_thr_test.pickle", "wb" ) )


#tweets with same hashtags
num_tweets_with_same_tags = dict()
for node in selected_users:
	num_tweets_with_same_tags[node] = sum([num_tweets_per_tag[x] for x in tags_for_user[node]])
	
with open("featuresUserSubset_test.csv","w") as fd:
	for i in selected_users_list:
		fd.write(str(i)+","+str(num_common_friends_thr[i])+","+str(num_rec[i])+","+str(num_tweets_with_same_tags[i])+"\n")
		