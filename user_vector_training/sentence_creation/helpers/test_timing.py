#extract hashtag graphs from sequence of authors adopting a hashtag and find all paths in the graphs to write into corpus file as sentences for training using word2vec 
#same as sentence_hashtag_adoption, but without using multiple processes. also, file to write sentences for 'ff' hashtag
import datetime
import sys
import os
import cPickle as pickle
import random
from collections import defaultdict
# from recursive_getsizeof import total_size

min_tweets_sequence = 2 # minimum number of tweets on a hashtag to remove hashtags with only few tweets available for extracting context

#conditions for edges between tweets
time_diff_for_edge = 12*60*60
follower_following_cond = False
geography_cond = False

context_length = 5 #m/2, length of context (to one side) or length of paths (half of the length) to consider, excluding present vertex
min_context_length = 2 #minimum length of context or length of paths to consider
gamma = 1 #number of contexts or paths for a tweet in a sequence

#read index of each user out of 7697889 users from map file

m = dict()
fr = open("/twitterSimulations/graph/map.txt")
for line in fr:
	line = line.rstrip()
	u = line.split(' ')
	m[int(u[0])] = int(u[1])
fr.close()
print 'Map Read'

#read adoption sequence from dif_timeline1s file
adoption_sequence = pickle.load(open("hashtagAdoptionSequences_workingset.pickle","rb"))
"""
# prev = ""
# count = 0
# rem=dict()
adoption_sequence = dict()
with open('/twitterSimulations/timeline_data/dif_timeline1s', 'r') as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split('\t')
		tag = u[0]
		time = int(u[1])
		author = m[int(u[2])]
		
		#remove non-emergent hashtags, about 70800 removed, 8793155 remaining out of 8863950 hashtags in dif_timeline1s
		# if tag != prev and prev != "":
			# if(count >= 5):
				# rem[prev] = len(adoption_sequence[prev])
				# del adoption_sequence[prev]
			# count = 0
		# if time < 1395901801: # less than 5 tweets in 12 hours from 1395858601
			# count = count + 1
		# prev = tag
		
		try:
			adoption_sequence[tag].append((time,author))
		except KeyError:
			adoption_sequence[tag]=[(time,author)]

# if(count >= 5):
	# rem[prev] = len(adoption_sequence[prev])
	# del adoption_sequence[prev]

print len(adoption_sequence)#, len(rem)
# pickle.dump(rem,open('nonEmergentHashtags.pickle','wb'))
"""
print "timeline file read"
#location information files
#can use location combined by country in known_locations_country_us and known_locations1_country_us files
max_locations = 141 #change number of unique locations to 97 for country_us files
location_buckets = [-1] * 7697889
# location_buckets = dict() #map to -1 for users not in location files
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

# reading follower graph files
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

for i in range(0, 7697889):
	follower_adj[i] = set(follower_adj[i])

print 'Graph Set\n'

#initialise adjacency list
def init_adj_list(num_nodes):
	adj = [[]] * num_nodes
	for i in range(0, num_nodes):
		adj[i] = []
	return adj

#get all paths starting from a vertex using DFS on hashtag graph
#Reference: http://eddmann.com/posts/depth-first-search-and-breadth-first-search-in-python/
def dfs_paths(adj,start):
	# visited = set()
	paths = []
	stack = [(start, [start])]
	while stack:
		(vertex, path) = stack.pop()
		nbh = set(adj[vertex]) - set(path)
		# nbh = set(adj[vertex]) - visited # visit each vertex once
		if len(nbh)==0:
			paths.append(path)
		for next in nbh:
			stack.append((next, path + [next])) #instead of all possible paths from all vertices, get maximum length paths in the graph
			# visited.add(next)
	return paths

#sample paths to the right of vertex from hashtag graph
def sample_paths_one_side(adj,present_node):
	paths = []
	for i in xrange(0,gamma):
		path=[present_node]
		count=0
		while count<context_length: #change context length value for single side
			adjacent_nodes = adj[present_node]
			if adjacent_nodes!=[]:
				present_node=random.choice(adjacent_nodes) #randomly choose one of the neighbours of present node
				path.append(present_node)
				count+=1
			else:
				break
		paths.append(path)
	return paths
	
#sample paths to left and right of vertex from hashtag graph
def sample_paths_both_side(adj,rev_adj,start):
	paths = []
	present_node = start
	for i in xrange(0,gamma):
		#left
		path=[]
		count=0
		while count<context_length: #change context length value for single side
			adjacent_nodes = rev_adj[present_node]
			if adjacent_nodes!=[]:
				present_node=random.choice(adjacent_nodes) #randomly choose one of the neighbours of present node
				path.append(present_node)
				count+=1
			else:
				break
		path.reverse()
		path.append(start)
		#right
		count=0
		present_node = start
		while count<context_length: #change context length value for single side
			adjacent_nodes = adj[present_node]
			if adjacent_nodes!=[]:
				present_node=random.choice(adjacent_nodes) #randomly choose one of the neighbours of present node
				path.append(present_node)
				count+=1
			else:
				break
		paths.append(path)
	return paths
	
#get user ids from vertex ids in paths
def path_to_sentence(nodes,path):
	s=[]
	for i in path:
		_,author = nodes[i]
		s.append(str(author)) #type of str for author is needed for using join
	return s
"""
#separate hashtag segments from adoption sequence of a hashtag using maximum time difference allowed for edges for reducing length of sequence to consider for hashtag graph
def get_adoption_segments(sequence):
	first_tw_time,first_tw_author = sequence[0]
	prev_time = first_tw_time
	seg = [] #group of tweets or segment
	seg.append(sequence[0])
	segments=[] #group of segments
	for i in sequence[1:]:
		time,_ = i
		if time-prev_time>time_diff_for_edge:
			segments.append(seg)
			seg = []
		seg.append(i)
		prev_time = time
	if seg!=[]:
		segments.append(seg)
	return segments
"""		
#get adjacency list of hashtag graph from a segment
def get_hashtag_graph_adj(segment):
	num_nodes = len(segment)
	# adj_list = init_adj_list(num_nodes) #adjacency list for directed graph
	adj_list = [[] for i in xrange(0, num_nodes)]
	rev_adj_list = [[] for i in xrange(0, num_nodes)] #defaultdict(list)
	# print "init", total_size(adj_list), total_size(rev_adj_list)
	# print "adj list init"
	if num_nodes==1:
		return adj_list, rev_adj_list
	location = [[] for i in xrange(0, max_locations)] #dict()
	for i in xrange(0,num_nodes):
		_,author = segment[i]
		author_loc = location_buckets[author]
		if author_loc!=-1: #no edges between users with unknown location
			location[author_loc].append(i) #time sorted order will change across locations, but not within location. order of vertices in adjacency list is still same
	# print "location list"
	count=0
	for same_loc_seq in location:
		num_loc = len(same_loc_seq)
		print count, "Count", len(same_loc_seq)
		count+=1
		for i in xrange(0,num_loc):
			vertex_index_first = same_loc_seq[i]
			time_first,author_first = segment[vertex_index_first]
			followers_author_first = follower_adj[author_first]
			for j in xrange(i+1,num_loc):
				vertex_index_second = same_loc_seq[j]
				time_second,author_second = segment[vertex_index_second]
				if time_second-time_first<=time_diff_for_edge: # only time difference considered for an edge, check other conditions
					if author_second in followers_author_first:
						adj_list[vertex_index_first].append(vertex_index_second)
						rev_adj_list[vertex_index_second].append(vertex_index_first)
					# rev_adj_list[vertex_index_second].insert(0,vertex_index_first) #to make the order of vertices having edge to second vertex in decreasing order, i.e., closest vertex first
				else:
					break #tweets are arranged in increasing time, so no edges will be there with vertices past present node
				#follower relation
				#check if more than one connected components in a segment if single path is considered for each segment
	# print "assigned", total_size(adj_list), total_size(rev_adj_list)

	return adj_list, rev_adj_list
"""
def get_hashtag_graph_adj(segment):
	num_nodes = len(segment)
	adj_list = [[] for i in xrange(0, num_nodes)]
	rev_adj_list = [[] for i in xrange(0, num_nodes)]
	print "adj list init"
	if num_nodes==1:
		return adj_list
	for i in xrange(0,num_nodes):
		time_first,author_first = segment[i]
		for j in xrange(i+1,num_nodes):
			time_second,author_second = segment[j]
			if time_second-time_first<=time_diff_for_edge: # only time difference considered for an edge, check other conditions
				if location_buckets[author_first]!=-1 and location_buckets[author_first]==location_buckets[author_second]:
					adj_list[i].append(j)
					rev_adj_list[j].append(i)
			else:
				break #tweets are arranged in increasing time, so no edges will be there with vertices past present node
			#follower relation
			#check if more than one connected components in a segment if single path is considered for each segment
	return adj_list, rev_adj_list
"""	
#get all paths or sample of paths of length, m (same as or double of context length) from hashtag graph
def get_paths_from_graph(nodes, adj, rev_adj):
	# paths = []
	if len(nodes)>=min_context_length: #only if less than m length paths are not taken
		# return []
		for start in xrange(0,len(nodes)):
			# if len(nodes)-start-1<min_context_length: #number of vertices left are less than min context length
				# break
				
			#DFS for paths starting from a vertex
			# paths_vertices = dfs_paths(adj,start)
			
			#sample paths from right of all nodes
			# paths_vertices = sample_paths_one_side(adj,start)
			
			#sample paths from left and right of all nodes
			paths_vertices = sample_paths_both_side(adj,rev_adj,start) #first find path to the left of present node
			
			for p in paths_vertices:
				if len(p)>=min_context_length: #only take paths above minimum context length
					# paths.append(path_to_sentence(nodes,p))
					yield path_to_sentence(nodes,p)
	# return paths

#get sentences from hashtag sequences
def get_sentences(adoption_sequence):
	tag_count = 0
	for t in adoption_sequence:
		# if t!='ff': # count 4103630, ff has 1081979 tweets, adj. list problem
			# continue
		segment=adoption_sequence[t]

		#print number of the hashtag being processed
		tag_count+=1
		# if tag_count%1000==0:
		print "Hashtag count", tag_count, "Hashtag", t, "tweets", len(segment)
		
		"""
		segments = get_adoption_segments(seq)
		for seg in segments:
			hashtag_graph_adj = get_hashtag_graph_adj(seg)
			paths = get_paths_from_graph(seg, hashtag_graph_adj)
			for p in paths: #change if only one path generated from a hashtag graph
				yield p 
		"""
		adj_list, rev_adj_list = get_hashtag_graph_adj(segment)
		print "Adjacency list formed"
		# paths = get_paths_from_graph(segment, adj_list, rev_adj_list)
		# print "Paths formed"
		
		del adj_list #memory not freed after function return
		del rev_adj_list
		
		if tag_count>2:
			break
		# for p in paths: #change if only one path generated from a hashtag graph
			# yield p 
			
#check how many users from dif_timeline1s are not mapped to any location
"""
not_found=set()
a=0
for t in adoption_sequence:
	seq=adoption_sequence[t]
	for time,author in seq:
		if location_buckets[author]==-1:
			not_found.add(author)
			a+=1
print "location unknown", len(not_found) #239476
print a #not unique count 13736074
"""
#check if adoption sequence is time sorted
"""
for t in adoption_sequence:
	seq=adoption_sequence[t]
	time_sorted = sorted(seq,key=lambda x: x[0]) # time is of type int
	if seq!=time_sorted:
		print "not time ordered", t #yes
"""
#write adoption sequence to file, from dif_timeline1s file remove hashtags with number of tweets less than min_tweets_sequence and map users to indices
"""
with open('hashtagAdoptionSequences.txt','wb') as fd:
	for tag in adoption_sequence.keys():
		if len(adoption_sequence[tag])>=min_tweets_sequence:
			# del adoption_sequence[tag]
			fd.write(tag)
			for t,a in adoption_sequence[tag]:
				fd.write(' '+str(t)+','+str(a)) #author is of type str for using join
			fd.write('\n')
	# pickle.dump(adoption_sequence,fd)
"""
#count number of adopters of each hashtag who are among the followers of present adopters of the hashtag at the time of adoption
"""
tagcount=0
following_adopters = []
total_tweet = []
adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt"
with open(adoption_sequence_filename, 'r') as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		adopter_followers = set()
		following_adopters_tag=0
		total_tweet_tag=0
		for i in range(1, len(u)):
			# timestamp = int(u[i][0:u[i].index(',')])
			total_tweet_tag+=1
			author = int(u[i][u[i].index(',')+1 : ])
			if author in adopter_followers:
				following_adopters_tag+=1
			adopter_followers.update(follower_adj[author])
		following_adopters.append(following_adopters_tag)
		total_tweet.append(total_tweet_tag)
		tagcount+=1
		if tagcount%100000==0:
			print "Hashtag count", tagcount
with open("following_adopters.pickle","wb") as fd:
	pickle.dump(following_adopters,fd)
	pickle.dump(total_tweet,fd)
print "Sequence file read"
"""
#write sentences to file
start_time = datetime.datetime.now()
get_sentences(adoption_sequence)
print "Start time", start_time, "End time", datetime.datetime.now()
"""
# count=defaultdict(int)
with open("hashtagAdoptionSentences_ff.txt","wb") as fd:
	start_time = datetime.datetime.now()
	sentences = get_sentences(adoption_sequence)
	for s in sentences:
		fd.write(" ".join(s)+"\n")
		# count[len(s)]+=1
		# print "Path length", len(s)
		# _=len(s)
	print start_time, datetime.datetime.now()
# pickle.dump(count,open("frequencyContextLength.pickle","wb"))
"""
