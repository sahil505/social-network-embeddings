# file to plot frequency distribution using matplotlib

import numpy as np
import matplotlib.pyplot as plt

rec_links = []
same_tags_tweets = []
with open("featuresUserSubset.csv","rb") as fr:
	for line in fr:
		line = line.rstrip()
		u = line.split(',')
		id,rec,tags = int(u[0]),int(u[1]),int(u[2])
		rec_links.append(rec)
		same_tags_tweets.append(tags)
rec_links = np.array(rec_links)
same_tags_tweets = np.array(same_tags_tweets)

num_bin = 100000
def freq_plot(data,xlab):
	values, base = np.histogram(data, bins=num_bin)
	cumulative = np.cumsum(values)
	plt.plot(base[:-1], values, c='red') #frequency
	# plt.plot(base[:-1], cumulative/float(len(data)), c='red') #normalised
	# plt.plot(base[:-1], len(data)-cumulative, c='red') #inverse, greater than
	# plt.plot(base[:-1], len(data)-np.append(0,cumulative)[:-1], c='red') #inverse, greater than or equal to
	# plt.yscale('log')
	plt.xscale('log')
	plt.xlabel(xlab)
	plt.xlim(xmin=0)
	plt.ylabel('cumulative frequency')
	plt.title('cumulative frequency distribution (greater than or equal to)')
	plt.grid()
	plt.show()

freq_plot(rec_links,'Users with reciprocal links')
freq_plot(same_tags_tweets,'Users with tweets on same hashtags')