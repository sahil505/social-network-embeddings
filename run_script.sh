## Basic workflow

cd /dbresearch2/word2vec/degree_distribution

### pre-process hashtag adoption sequences ###
# filter users
python "deg_dist_in_top_users.py"

# create hashtag sequences. Output to sentences_files_timeonly/
python "filter_hashtag_sequence.py"

# filter follower graph
python "filter_follower_graph.py"

# get hashtags tweeted by these users
python "train_test_split_hashtag_sequence.py"

### corpus creation ###
# convert to sentences
python "sentence_creation/sentence_hashtag_adoption.py"

# concatenate sentence files. Output userSentencesComb file.
bash cat_files.sh

### train vectors using word2vec ###
# word2vec to get user vectors. Output node_vectors_1hr_bfsr.txt and node_vocab_1hr_bfsr.txt
bash "node_vector_training.sh"

### adopter prediction task ###
# frequency and exposure rank baselines
python "adopter_prediction_baseline.py"

# user vector averaging method
python "adopter_prediction_multiple_prec_plot.py"

### geolocation prediction task ###
# classification method and baselines
python "user_vector_cluster_geography.py"