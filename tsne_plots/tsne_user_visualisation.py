#visualising users who adopted a hashtag using t-SNE on user vectors

import matplotlib
matplotlib.use('Agg')
from tsne import *
from numpy import array
import math, random

word_vectors = []
path_vec_file = '/mnt/filer01/word2vec/node_vectors_1hr_pr.txt'
word_vector_dim = 100
labels = dict()
X_word = []
windex=0
with open(path_vec_file, 'rb') as fr:
	_,dim = next(fr).rstrip().split(' ')
	word_vector_dim = int(dim)
	next(fr)
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		if len(u) != word_vector_dim+1:
			print "vector length error"
		word = int(u[0])
		vec = map(float,u[1:])
		labels[word]=windex
		windex+=1
		X_word.append(vec)

word_freq_sorted = []
path_vocab_file = '/mnt/filer01/word2vec/node_vocab_1hr_pr.txt'
with open(path_vocab_file, 'rb') as fr:
	next(fr)
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		word_freq_sorted.append(int(u[0]))

tag_seq = dict()
users_ht = set()
seq_file = '/mnt/filer01/word2vec/degree_distribution/sample_ht_sequences'
with open(seq_file, 'rb') as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		tag = u[0]
		sequence = []
		for i in range(1, len(u)):
			author = int(u[i][u[i].index(',')+1 : ])
			if author in labels:
				sequence.append(author)
		tag_seq[tag]=random.sample(sequence,500)
		for u in tag_seq[tag]:
			users_ht.add(u)
		print tag, len(sequence)
print len(users_ht)
		
m = dict()
fr = open("/twitterSimulations/graph/map.txt")
for line in fr:
	line = line.rstrip()
	u = line.split(' ')
	m[int(u[0])] = int(u[1])
fr.close()
print 'Map Read'

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

def get_word_vectors_ht(wlist):
	vectors = []
	color = []
	tags=tag_seq.keys()
	c1 = set(tag_seq[tags[0]]) #modikiadalat (dark blue)
	c2 = set(tag_seq[tags[1]]) #7millionandcounting (light blue)
	c3 = set(tag_seq[tags[2]]) #time100 (red)
	for w in wlist:
		vectors.append(X_word[labels[w]])
		if w in c1:
			color.append(50)
		elif w in c2:
			color.append(100)
		elif w in c3:
			color.append(200)
		else:
			print "no tag"
	return array(vectors), color

def get_word_vectors(wlist):
	vectors = []
	color = []
	for w in wlist:
		vectors.append(X_word[labels[w]])
		color.append(location_buckets[w])
	return array(vectors), color
	
# most_freq = word_freq_sorted[0:2500]
# least_freq = word_freq_sorted[-2500:]
half_num_words = int(len(word_freq_sorted)/2.0)
mid_freq = word_freq_sorted[half_num_words-1250:half_num_words+1249]
all_random = random.sample(word_freq_sorted,1000)

def save_embed_plot((X,color),fname):
	Y = tsne(X, no_dims = 2, initial_dims = 50, perplexity = 30.0);
	fig = Plot.figure()
	Plot.scatter(Y[:,0], Y[:,1], s=20, c=color, alpha=0.8, edgecolor='none');
	Plot.axis('off')
	fig.savefig(fname, dpi=300, bbox_inches='tight')
	
if __name__ == "__main__":
	print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
	# print "Running example on 2,500 MNIST digits..."
	# X = Math.loadtxt("mnist2500_X.txt");
	# labels = Math.loadtxt("mnist2500_labels.txt");
	# save_embed_plot(get_word_vectors(most_freq),'embed_users_mostfreq.png')
	save_embed_plot(get_word_vectors(all_random),'embed_users_random_1hr_pr.png')
	# save_embed_plot(get_word_vectors(mid_freq),'embed_users_midfreq_1hr_pr.png')
	save_embed_plot(get_word_vectors_ht(list(users_ht)),'embed_users_random_ht_1hr_pr.png')
	# save_embed_plot(get_word_vectors(least_freq),'embed_users_leastfreq.png')