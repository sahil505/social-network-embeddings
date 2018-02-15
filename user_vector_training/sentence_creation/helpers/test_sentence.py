#extract hashtag graphs from sequence of authors adopting a hashtag and find all paths in the graphs to write into corpus file as sentences for training using word2vec 
import time
import sys
import os
import cPickle as pickle
import random

min_tweets_sequence = 0 # minimum number of tweets on a hashtag to remove hashtags with only few tweets available for extracting context

#conditions for edges between tweets
time_diff_for_edge = 10
# time_diff_for_edge = 12*60*60
follower_following_cond = False
geography_cond = False

context_length = 4 #m/2, length of context (to one side) or length of paths (half of the length) to consider
min_context_length = 0 #minimum length of context or length of paths to consider
gamma = 1 #number of contexts or paths for a tweet in a sequence
NUM_LEVEL_LIMIT = 2

def get_location(author):
	if author in location_buckets:
		return location_buckets[author]
	else:
		return -1 #location unknown
		
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
	print "Start", start
	for i in xrange(0,gamma):
		#left
		path=[]
		count=0
		while count<context_length: #change context length value for single side
			adjacent_nodes = rev_adj[present_node]
			if adjacent_nodes!=[]:
				present_node=random.choice(adjacent_nodes) #randomly choose one of the neighbours of present node
				path.append(present_node)
				print present_node, "out of", adjacent_nodes, "left"
				count+=1
			else:
				break
		path.reverse()
		path.append(start)
		print path, "from left"
		#right
		count=0
		present_node = start
		while count<context_length: #change context length value for single side
			adjacent_nodes = adj[present_node]
			if adjacent_nodes!=[]:
				present_node=random.choice(adjacent_nodes) #randomly choose one of the neighbours of present node
				path.append(present_node)
				print present_node, "out of", adjacent_nodes, "right"
				count+=1
			else:
				break
		paths.append(path)
		print path, "full"
	return paths

#sample neighbouring vertices right of a vertex from hashtag graph in bfs manner
def sample_nbhs_bfs_one_side(adj,start):
	paths = []
	for i in xrange(0,gamma):
		print "Start", start
		present_node=start
		path=[present_node]
		count=0
		next_level=adj[present_node]
		visited=set()
		visited.add(present_node)
		num_levels=1
		while count<context_length and next_level!=[] and num_levels<=NUM_LEVEL_LIMIT: #change context length value for single side
			num_remaining = context_length-count
			if len(next_level) < num_remaining:
				print path, next_level, "before", num_remaining
				path+=next_level
				count+=len(next_level)
				visited.update(next_level)
				nbh=set()
				for vertex_p in next_level:
					for vertex_n in adj[vertex_p]:
						if vertex_n not in visited:
							nbh.add(vertex_n)
				next_level=list(nbh)
				num_levels+=1
				print path, next_level, "after", visited
			elif len(next_level) > num_remaining:
				path+=random.sample(next_level,num_remaining) #order of vertices changed by sample
				print path, "out of", next_level
				break
			else:
				path+=next_level
				print path, next_level, "all"
				break
		paths.append(path)
	return paths
	
#sample neighbouring vertices to left and right of vertex from hashtag graph
def sample_nbhs_bfs(adj,rev_adj,start):
	paths = []
	for i in xrange(0,gamma):
		#left
		path=[]
		count=0
		queue=[start]
		visited=set()
		while count<context_length+1 and queue!=[]: #change context length value for single side
			present_node = queue.pop()
			if present_node not in visited:
				# present_node=random.choice(adjacent_nodes) #randomly choose one of the neighbours of present node
				visited.add(present_node)
				adjacent_nodes = [node for node in rev_adj[present_node] if node not in visited]
				path=[present_node]+path
				queue=adjacent_nodes+queue
				count+=1
		path=path[:-1]
		#right
		count=0
		queue=[start]
		visited=set()
		while count<context_length+1 and queue!=[]: #change context length value for single side
			present_node = queue.pop(0)
			if present_node not in visited:
				# present_node=random.choice(adjacent_nodes) #randomly choose one of the neighbours of present node
				visited.add(present_node)
				adjacent_nodes = [node for node in adj[present_node] if node not in visited]
				path.append(present_node)
				queue+=adjacent_nodes
				count+=1
		paths.append(path)
	return paths
	
#get user ids from vertex ids in paths
def path_to_sentence(nodes,path):
	s=[]
	for i in path:
		_,author = nodes[i]
		s.append(str(author)) #type of str for author is needed for using join
	return s
	
#separate hashtag segments from adoption sequence of a hashtag using maximum time difference allowed for edges for reducing length of sequence to consider for hashtag graph
"""
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
"""
def get_hashtag_graph_adj(segment):
	num_nodes = len(segment)
	adj_list = init_adj_list(num_nodes) #adjacency list for directed graph
	if num_nodes==1:
		return adj_list
	for i in range(0,num_nodes):
		time_first,_ = segment[i]
		for j in range(i+1,num_nodes):
			time_second,_ = segment[j]
			if time_second-time_first<=time_diff_for_edge: # only time difference considered for an edge, check other conditions
				adj_list[i].append(j)
			else:
				break #tweets are arranged in increasing time, so no edges will be there with vertices past present node
			#location
			#follower relation
			#check if more than one connected components in a segment if single path is considered for each segment
	return adj_list
"""

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
	print "location list", location
	count=0
	for same_loc_seq in location:
		num_loc = len(same_loc_seq)
		print count, "Count", len(same_loc_seq)
		count+=1
		for i in xrange(0,num_loc):
			vertex_index_first = same_loc_seq[i]
			time_first,_ = segment[vertex_index_first]
			for j in xrange(i+1,num_loc):
				vertex_index_second = same_loc_seq[j]
				time_second,_ = segment[vertex_index_second]
				if time_second-time_first<=time_diff_for_edge: # only time difference considered for an edge, check other conditions
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
#get adjacency list of hashtag graph from a segment, using only time diff
def get_hashtag_graph_adj(segment):
	num_nodes = len(segment)
	adj_list = [[] for i in xrange(0, num_nodes)]
	rev_adj_list = [[] for i in xrange(0, num_nodes)]
	# print "adj list init"
	if num_nodes==1:
		return adj_list, rev_adj_list
	for i in xrange(0,num_nodes):
		time_first,author_first = segment[i]
		for j in xrange(i+1,num_nodes):
			time_second,author_second = segment[j]
			if time_second-time_first<=time_diff_for_edge: # only time difference considered for an edge, check other conditions
				adj_list[i].append(j)
				rev_adj_list[j].append(i)
			else:
				break #tweets are arranged in increasing time, so no edges will be there with vertices past present node
			#follower relation
			#check if more than one connected components in a segment if single path is considered for each segment
	return adj_list, rev_adj_list
"""
#get all paths of length m from hashtag graph
def get_paths_from_graph(nodes, adj, rev_adj):
	if len(nodes)>=min_context_length: #only if less than m length paths are not taken
		for start in xrange(0,len(nodes)):
			# if len(nodes)-start-1<min_context_length: #number of vertices left are less than min context length
				# break
				
			#DFS for paths starting from a vertex
			# paths_vertices = dfs_paths(adj,start)
			
			#sample paths from right of all nodes
			# paths_vertices = sample_paths_one_side(adj,start)
			
			#sample paths from left and right of all nodes
			# paths_vertices = sample_paths_both_side(adj,rev_adj,start) #first find path to the left of present node
			
			#sample neighbours from left and right of all nodes in breadth-first search way
			# paths_vertices = sample_nbhs_bfs(adj,rev_adj,start)
			
			#sample neighbours from right of all nodes in breadth-first search way
			paths_vertices = sample_nbhs_bfs_one_side(adj,start)
			
			for p in paths_vertices:
				if len(p)>=min_context_length: #only take paths above minimum context length
					yield (start,path_to_sentence(nodes,p))

#get sentences from hashtag sequences
sentences=[]
max_locations = 2
adoption_sequence = dict()
adoption_sequence['test']=[(4,0),(10,1),(15,2),(21,3),(23,4),(26,5),(28,6),(37,7),(40,8),(45,9)]
location_buckets = [0,0,1,1,-1,1,1,1,1,1]

def get_sentences(adoption_sequence):
	tag_count = 0
	for t in adoption_sequence:
		segment=adoption_sequence[t]
		tag_count+=1
		adj_list, rev_adj_list = get_hashtag_graph_adj(segment)
		print adj_list, rev_adj_list
		paths = get_paths_from_graph(segment, adj_list, rev_adj_list)
		for p in paths: #change if only one path generated from a hashtag graph
			yield p

print(list(get_sentences(adoption_sequence)))