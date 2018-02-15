#visualising top 100 most, least and mid frequent hashtags using t-SNE on histogram-of-counts vectors from word classes

import matplotlib
matplotlib.use('Agg')
from tsne import *
from numpy import array
import math, random
import cPickle as pickle
from collections import Counter

path_class_file = '/mnt/filer01/word2vec/twitter_vectors_classes.sorted.txt'
word_to_cluster = dict()
with open(path_class_file, 'rb') as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		word_to_cluster[u[0]]=int(u[1])

tag_labels = []
num_tags = 0
with open('/mnt/filer01/tweets_repository/Nov2013/tag_tweets_bow.txt', 'rb') as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split('\t')
		tag = u[0]
		tag_labels.append(tag)
		num_tags+=1

word_doc_freq = dict()
tag_bow = []
with open('/mnt/filer01/tweets_repository/Nov2013/tag_tweets_bow_processed.txt', 'rb') as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		tag = u[0]
		words = u[1:]
		tag_bow.append(words) # remove duplicate words also
		doc_words = set()
		for w in words:
			if w not in doc_words:
				if w not in word_doc_freq:
					word_doc_freq[w]=0
				word_doc_freq[w]+=1
				doc_words.add(w)
		
word_clusters_dim = 1000
word_not_found=set()
hist_feature = []
for tag_words in tag_bow:
	tag_feature = [0]*word_clusters_dim
	num_words = 0
	# word_term_freq = Counter(tag_words)
	for word in tag_words:
		try: #words from tag bow missing in word vector, may be because of min limit on word occurrence 5
			cluster_id = word_to_cluster[word] # cluster index from 0, and order of idx and labels same
			df = word_doc_freq[word] #document frequency of words from vocab file
			idf = math.log10(float(num_tags)/df)
			# if word=='dconcert': # count for cluster with 'dconcert' very high, causing nan value error in tsne P-value calculation
				# continue
			tag_feature[cluster_id]+=1*idf #using idf as word relevance
			num_words+=1*idf
		except:
			word_not_found.add(word)
	#normalise by total number of words
	# num_words = len(tag_words)
	if num_words==0:
		print "error, tag with no words"
		num_words = 0.1
	hist_feature.append([float(x)/num_words for x in tag_feature])
# with open('hashtag_vec_tfidf.pickle', 'wb') as fd:
	# pickle.dump(hist_feature,fd)
print len(word_doc_freq), len(word_not_found)
tag_freq = []
with open('tag_freq_1500.csv', 'rb') as fr:
	next(fr)
	for line in fr:
		line = line.rstrip()
		u = line.split(',')
		tag = u[0]
		tag_freq.append((tag,int(u[1])))

tag_freq_sorted = [t for t,_ in sorted(tag_freq,key=lambda x: x[1], reverse = True)]
most_freq = set(tag_freq_sorted[0:150])
least_freq = set(tag_freq_sorted[-150:])
half_num_words = int(len(tag_freq_sorted)/2.0)
mid_freq = set(tag_freq_sorted[half_num_words-75:half_num_words+74])
all_random = set(random.sample(tag_freq_sorted,150))

#set visibility of most, least and mid frequency hashtags by setting text size
def get_tag_size_label(tlist):
	size = []
	label = []
	for t in tag_labels:
		if t in tlist:
			size.append(2)
			label.append(t.decode('latin-1'))
		else:
			size.append(0)
			label.append('')
	return size, array(label)

X = array(hist_feature)
Y = tsne(X, 2, 50, 30.0);

def save_embed_plot((tag_sizes,labels),fname):
	fig = Plot.figure()
	Plot.scatter(Y[:,0], Y[:,1], 0);
	for label, x, y, s in zip(labels, Y[:,0], Y[:,1], tag_sizes):
		Plot.annotate(label, xy = (x, y), xytext = (0, 0), textcoords = 'offset points', size=s)
	Plot.axis('off')
	fig.savefig(fname, dpi=800, bbox_inches='tight')
	
if __name__ == "__main__":
	print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
	# print "Running example on 2,500 MNIST digits..."
	# X = Math.loadtxt("mnist2500_X.txt");
	# labels = Math.loadtxt("mnist2500_labels.txt");
	save_embed_plot(get_tag_size_label(most_freq),'embed_tag_mostfreq.png')
	save_embed_plot(get_tag_size_label(mid_freq),'embed_tag_midfreq.png')
	save_embed_plot(get_tag_size_label(least_freq),'embed_tag_leastfreq.png')
	save_embed_plot(get_tag_size_label(all_random),'embed_tag_random.png')