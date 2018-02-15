#extract hashtag graphs from sequence of authors adopting a hashtag and find all paths in the graphs to write into corpus file as sentences for training using word2vec 
import datetime
import sys
import os
import cPickle as pickle
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import traceback

min_tweets_sequence = 2 # minimum number of tweets on a hashtag to remove hashtags with only few tweets available for extracting context

#conditions for edges between tweets
time_diff_for_edge = 1*60*60
follower_following_cond = False
geography_cond = False

context_length = 10 #m/2, length of context (to one side) or length of paths (half of the length) to consider, excluding present vertex
min_context_length = 2 #minimum length of context or length of paths to consider
gamma = 1 #number of contexts or paths for a tweet in a sequence
NUM_LEVEL_LIMIT = 3
seq_len_threshold = 500 # minimum number of adopters for a hashtag to consider the sequence for sentences 

adoption_sequence_filename = "/mnt/filer01/word2vec/degree_distribution/hashtagAdoptionSequences.txt"
NUM_LINES = 3617312 #2701284 #number lines in adoption_sequence_filename
out_dir = "/mnt/filer01/word2vec/degree_distribution/sentences_files_loc/" #make output directory if it doesn't exist
NUM_PROCESSES = 4

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
# adoption_sequence = pickle.load(open("hashtagAdoptionSequences_workingset.pickle","rb"))
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
# print "timeline file read"

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
def sample_paths_one_side(adj,start):
	paths = []
	for i in xrange(0,gamma):
		present_node=start
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
	for i in xrange(0,gamma):
		#left
		present_node = start
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

#sample neighbouring vertices right of a vertex from hashtag graph in bfs manner
def sample_nbhs_bfs_one_side(adj,start):
	paths = []
	for i in xrange(0,gamma):
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
			elif len(next_level) > num_remaining:
				path+=random.sample(next_level,num_remaining) #order of vertices changed by sample
				break
			else:
				path+=next_level
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

#get adjacency list of hashtag graph from a segment, using only time diff
def get_hashtag_graph_adj(segment):
	num_nodes = len(segment)
	adj_list = [[] for i in xrange(0, num_nodes)]
	#rev_adj_list = [[] for i in xrange(0, num_nodes)]
	# print "adj list init"
	if num_nodes==1:
		return adj_list#, rev_adj_list
	for i in xrange(0,num_nodes):
		time_first,author_first = segment[i]
		for j in xrange(i+1,num_nodes):
			time_second,author_second = segment[j]
			if time_second-time_first<=time_diff_for_edge: # only time difference considered for an edge, check other conditions
				adj_list[i].append(j)
				#rev_adj_list[j].append(i)
			else:
				break #tweets are arranged in increasing time, so no edges will be there with vertices past present node
			#follower relation
			#check if more than one connected components in a segment if single path is considered for each segment
	return adj_list#, rev_adj_list

#get adjacency list of hashtag graph from a segment, using both geography and time difference
def get_hashtag_graph_adj_geo_time(segment):
	num_nodes = len(segment)
	#adjacency list for directed graph
	adj_list = [[] for i in xrange(0, num_nodes)]
	#rev_adj_list = [[] for i in xrange(0, num_nodes)] #defaultdict(list)
	# print "adj list init"
	if num_nodes==1:
		return adj_list#, rev_adj_list
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
		# print count, "Count", len(same_loc_seq)
		count+=1
		for i in xrange(0,num_loc):
			vertex_index_first = same_loc_seq[i]
			time_first,_ = segment[vertex_index_first]
			for j in xrange(i+1,num_loc):
				vertex_index_second = same_loc_seq[j]
				time_second,_ = segment[vertex_index_second]
				if time_second-time_first<=time_diff_for_edge: # only time difference considered for an edge, check other conditions
					adj_list[vertex_index_first].append(vertex_index_second)
					#rev_adj_list[vertex_index_second].append(vertex_index_first)
					# rev_adj_list[vertex_index_second].insert(0,vertex_index_first) #to make the order of vertices having edge to second vertex in decreasing order, i.e., closest vertex first
				else:
					break #tweets are arranged in increasing time, so no edges will be there with vertices past present node
				#follower relation
				#check if more than one connected components in a segment if single path is considered for each segment
	return adj_list#, rev_adj_list

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
def get_paths_from_graph(nodes, adj):
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
			# paths_vertices = sample_paths_both_side(adj,rev_adj,start) #first find path to the left of present node
			
			#sample neighbours from left and right of all nodes in breadth-first search way
			# paths_vertices = sample_nbhs_bfs(adj,rev_adj,start)
			
			#sample neighbours from right of all nodes in breadth-first search way
			paths_vertices = sample_nbhs_bfs_one_side(adj,start)
			
			for p in paths_vertices:
				if len(p)>=min_context_length: #only take paths above minimum context length
					# paths.append(path_to_sentence(nodes,p))
					yield path_to_sentence(nodes,p)
	# return paths

#read lines from adoption sequence file
def read_adoption_sequence(adoption_sequence_filename, start, end,train_seq_id,large_tag_id):
	with open(adoption_sequence_filename, 'r') as fr:
		count=0
		for line in fr:
			if count < start:
				count+=1
				continue
			elif count >= end:
				return
			if count not in train_seq_id or count in large_tag_id:
				count+=1
				continue
			count+=1
			line = line.rstrip()
			u = line.split(' ')
			tag = u[0]
			sequence = []
			adopters = set()
			for i in range(1, len(u)):
				timestamp = int(u[i][0:u[i].index(',')])
				author = int(u[i][u[i].index(',')+1 : ])
				sequence.append((timestamp,author))
				adopters.add(author)
			if len(adopters) < seq_len_threshold:
				continue
			yield (tag,sequence)
			
#get sentences from hashtag sequences read from input file from start to end line numbers
def get_sentences(adoption_sequence_filename,start_line_num,end_line_num,process_num,train_seq_id,large_tag_id):
	tag_count = 0
	for t,segment in read_adoption_sequence(adoption_sequence_filename,start_line_num,end_line_num,train_seq_id,large_tag_id):
		if t=='ff': # count 4103630, ff has 1081979 tweets, adj. list problem
			# continue
			print 'Hashtag ff'
		
		#print number of the hashtag being processed
		tag_count+=1
		if tag_count%10000==0:
			print "Process", process_num, "Hashtag count", tag_count, "Hashtag", t#, "tweets", len(segment)

		# adj_list = get_hashtag_graph_adj(segment)
		adj_list = get_hashtag_graph_adj_geo_time(segment)
		# print "Adjacency list formed"
		paths = get_paths_from_graph(segment, adj_list)
		# print "Paths formed"
		
		# del adj_list #memory not freed after function return
		# del rev_adj_list
		
		# if tag_count>2:
			# break
		for p in paths: #change if only one path generated from a hashtag graph
			yield p 
			
#check how many users from dif_timeline1s are not mapped to any location
"""
not_found=0
for t in adoption_sequence:
	seq=adoption_sequence[t]
	for time,author in seq:
		if author not in location_buckets:
			not_found+=1
print not_found #13736074
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
#write sentences to file
# get_sentences(adoption_sequence)

# count=defaultdict(int)

def write_sentence(process_num,start,end,train_seq_id,large_tag_id):
	try:	
		with open(out_dir+'/hashtagAdoptionSentences'+str(process_num)+'.txt','wb') as fd:
			start_time = datetime.datetime.now()
			sentences = get_sentences(adoption_sequence_filename,start,end,process_num,train_seq_id,large_tag_id)
			for s in sentences:
				fd.write(" ".join(s)+"\n")
				# count[len(s)]+=1
			print "Process", process_num, "Hashtags", start, "to", end-1, "Start time", start_time, "End time", datetime.datetime.now()
		return process_num,0
	except Exception as e:
		print traceback.format_exc()
		return process_num,1
		#sys.exit(1)
# pickle.dump(count,open("frequencyContextLength.pickle","wb"))

#indices of lines in sequence file for training
with open("sequence_file_split_indices.pickle","rb") as fr:
	train_seq_id = pickle.load(fr)
train_seq_id = set(train_seq_id)

#indices of lines in sequence file with large sequences
# with open("sequence_large_hashtags.pickle","rb") as fr:
	# large_tag_id = pickle.load(fr)
# large_tag_id = set(large_tag_id)&train_seq_id
large_tag_id = []

# write_sentence('',0,2)
#run write_sentences on different chunks of adoption sequence file in parallel processes
num_workers = min(NUM_PROCESSES,cpu_count())
pool = Pool(processes=num_workers) 
process_num=0
lines_per_process = int(NUM_LINES/(2.0*num_workers))
for s,e in ( (i,min(i+lines_per_process,NUM_LINES)) for i in xrange(0,NUM_LINES,lines_per_process) ):
	pool.apply_async(write_sentence, args=(process_num,s,e,train_seq_id,large_tag_id))
	process_num+=1
pool.close()
pool.join()
"""
process_num=13
#run write_sentences on hashtags with large sequences in adoption sequence file
with open(out_dir+'/hashtagAdoptionSentences'+str(process_num)+'.txt','wb') as fd:
	start_time = datetime.datetime.now()
	tag_count = 0
	large_adopt_seq = dict()
	tag_length = []
	for t,segment in read_adoption_sequence(adoption_sequence_filename,0,NUM_LINES,large_tag_id,[]):
		large_adopt_seq[t]=segment
		tag_length.append((t,len(segment)))
	tag_length_sorted = sorted(tag_length,key=lambda x: x[1])
	for t,_ in tag_length_sorted:
		# if t=='ff': # count 4103630, ff has 1081979 tweets, adj. list problem
			# continue
			# print 'Hashtag ff'
		segment = large_adopt_seq[t]
		#print number of the hashtag being processed
		tag_count+=1
		print "Process", process_num, "Hashtag count", tag_count, "Hashtag", t, "tweets", len(segment)

		# adj_list = get_hashtag_graph_adj(segment)
		# print "Adjacency list formed"
		# paths = get_paths_from_graph(segment, adj_list)
		# print "Paths formed"
		
		# del adj_list #memory not freed after function return
		# del rev_adj_list
		
		# if tag_count>2:
			# break
		# for p in paths: #change if only one path generated from a hashtag graph
			# fd.write(" ".join(p)+"\n")
		segments = get_adoption_segments(segment)
		for seg in segments:
			hashtag_graph_adj = get_hashtag_graph_adj(seg)
			paths = get_paths_from_graph(seg, hashtag_graph_adj)
			for p in paths: #change if only one path generated from a hashtag graph
				fd.write(" ".join(p)+"\n")
			del hashtag_graph_adj
print "Process", process_num, "large hashtags", "Start time", start_time, "End time", datetime.datetime.now()
"""