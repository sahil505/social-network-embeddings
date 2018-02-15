# social-network-embedding
Code for methods to embed social network users based on their topic activity, described in the [paper](https://arxiv.org/abs/1710.07622)

```
Learning user representations in Online Social Networks using temporal dynamics of information diffusion.
Harvineet Singh, Amitabha Bagchi, and Parag Singla.
arXiv:1710.07622 cs.SI
```

#### Abstract

```
This article presents a novel approach for learning low-dimensional distributed representations of users in online social networks. Existing methods rely on the network structure formed by the social relationships among users to extract these representations. 
However, the network information can be obsolete, incomplete or dynamically changing. In addition, in some cases, it can be prohibitively expensive to get the network information. Therefore, we propose an alternative approach based on observations from topics being talked on in social networks. 
We utilise the time information of users adopting topics in order to embed them in a real-valued vector space. Through extensive experiments, we investigate the properties of the representations learned and their efficacy in preserving information about link structure among users. 
We also evaluate the representations in two different prediction tasks, namely, predicting most likely future adopters of a topic and predicting the geo-location of users. Experiments to validate the proposed methods are performed on a large-scale social network extracted from Twitter, consisting of about 7.7 million users and their activity on around 3.6 million topics over a month-long period.
```

#### Note
Adventurers beware! This repository is meant for version control of scripts used for experiments in the paper. So, not heavily commented and not heavily tested.

## Basic workflow
1. "user_vector_training/deg_dist_in_top_users.py" -> filter users
2. "user_vector_training/filter_hashtag_sequence.py" -> get hashtags tweeted by these users
3. "user_vector_training/sentence_creation/sentence_hashtag_adoption.py" -> convert to sentences
4. "user_vector_training/word2vec/twitter_training_w2v.sh" -> word2vec to get user vectors
5. "adopter_prediction/adopter_prediction.py" -> next adopter prediction

## Contact
If you are interested in knowing more or have any questions on the code, feel free to contact me at <harvineet1992@gmail.com>.