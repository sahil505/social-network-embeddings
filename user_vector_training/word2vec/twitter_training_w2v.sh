make

CORPUS=tweets_comb_processed
SAVE_FILE=twitter_vectors.txt
VOCAB_FILE=vocab.txt

time ./word2vec -train $CORPUS -output $SAVE_FILE -cbow 0 -size 200 -window 8 -negative 5 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15 -save-vocab $VOCAB_FILE
