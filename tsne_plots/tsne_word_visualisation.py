#visualising top 100 most, least and mid frequent words using t-SNE

from tsne import *
from numpy import array

word_vectors = []
path_vec_file = '/mnt/filer01/word2vec/twitter_vectors.txt'
word_vector_dim = 200
labels = dict()
X_word = []
windex=0
with open(path_vec_file, 'rb') as fr:
	next(fr)
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		if len(u) != word_vector_dim+1:
			print "vector length error"
		word = u[0].decode('latin-1')
		vec = map(float,u[1:])
		labels[word]=windex
		# word_vectors.append([word]+vec)
		X_word.append(vec)
		windex+=1
# labels = [x[0] for x in word_vectors]
# X_word = [x[1:] for x in word_vectors]

word_freq_sorted = []
path_vocab_file = '/mnt/filer01/word2vec/vocab.txt'
with open(path_vocab_file, 'rb') as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(' ')
		word_freq_sorted.append(u[0].decode('latin-1'))

def get_word_vectors(wlist):
	vectors = []
	for w in wlist:
		vectors.append(X_word[labels[w]])
	return array(vectors)
		
most_freq = word_freq_sorted[0:1000]
mid_freq = word_freq_sorted[-1000:]
half_num_words = int(len(word_freq_sorted)/2.0)
least_freq = word_freq_sorted[half_num_words-500:half_num_words+499]

def save_embed_plot(X,labels,fname):
	Y = tsne(X, 2, word_vector_dim, 20.0);
	fig = Plot.figure()
	Plot.scatter(Y[:,0], Y[:,1], 1);
	for label, x, y in zip(labels, Y[:,0], Y[:,1]):
		Plot.annotate(label, xy = (x, y), xytext = (0, 0), textcoords = 'offset points', size=5)
	fig.savefig(fname, dpi=1200)
	
if __name__ == "__main__":
	print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
	# print "Running example on 2,500 MNIST digits..."
	# X = Math.loadtxt("mnist2500_X.txt");
	# labels = Math.loadtxt("mnist2500_labels.txt");
	
	# save_embed_plot(get_word_vectors(most_freq),array(most_freq),'embed_mostfreq.png')
	# save_embed_plot(get_word_vectors(mid_freq),array(mid_freq),'embed_midfreq.png')
	# save_embed_plot(get_word_vectors(least_freq),array(least_freq),'embed_leastfreq.png')
	save_embed_plot(get_word_vectors(word_freq_sorted),array(word_freq_sorted),'embed_all.png')
	