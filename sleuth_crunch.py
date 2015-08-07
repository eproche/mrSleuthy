import sys
import logging
import numpy as np
from numpy import *
from sklearn.metrics.pairwise import cosine_similarity

def vocab_build(corpus):
	"""Builds set of unique words """
	lexicon = set()
	for doc in corpus:
		doc = doc.split()
		lexicon.update([word for word in doc])
	return lexicon

def attach_tfidf_weights(storage, vocab, tf_arr):
	"""Appends tf-idf weights to each word """
	wordlist = vocab
	storage_weighted = []
	for i in range(len(storage)):
		sys.stdout.write(str(i)+",")
		sys.stdout.flush()
		docweights = []
		stor_list = storage[i].split()
		for word in stor_list:
			words = [word,0]
			for j in range(len(wordlist)):
				if (wordlist[j] == word):
					words[1] = tf_arr[i][j]	
			docweights.append(words)
		storage_weighted.append(docweights)
	return storage_weighted

def featureVec(storage, model, num_features):
	"""creates a vector representation for each image's descriptions (document)
	"""
	index2word_set = set(model.index2word)
	realWords = [] #in model
	notfound = [] 
	feature_vecs = []
	tot_wei = 0.0 #tf-idf weight total
	for i in range(len(storage)):
		realWords.append([])
		for word in storage[i]:
			#cap = word[0].capitalize() catch if capitalized proper noun in model
			#word[0] = "/en/"+word[0] if using freebase_skipgram_1000.bin.gz
			if (word[0] in index2word_set):
				realWords[i].append(word)
				tot_wei += word[1]
				continue
	print tot_wei
	for i in range(len(realWords)):
		feature_vec = np.zeros((num_features), dtype="float32")
		num_words = 0
		for realword in realWords[i]:
			weighted_vec = model[realword[0]]*(realword[1] / tot_wei) #normalized tf-idf weight
			feature_vec = np.add(feature_vec, weighted_vec)
			num_words += 1
		feature_vec = np.divide(feature_vec, num_words) #average of each word vector
		feature_vecs.append(feature_vec)
	return feature_vecs

def featureVec_unweighted(storage, model, num_features):
	""" Same as featureVec, but no tf-idf weights"""
	index2word_set = set(model.index2word)
	realWords = []
	feature_vecs = []
	for i in range(len(storage)):
		realWords.append([])
		storage[i] = storage[i].split()
		for word in storage[i]:
			#word = "/en/"+word if using freebase_skipgram_1000.bin.gz
			if word in index2word_set:
				realWords[i].append(word)
			else:
				click.secho("notfound:  ", fg='red')
				click.echo(word)
	for i in range(len(realWords)):
		feature_vec = np.zeros((num_features), dtype="float32")
		num_words = 0
		for realword in realWords[i]:
			weighted_vec = model[realword]
			feature_vec = np.add(feature_vec, weighted_vec)
			num_words += 1
		feature_vec = np.divide(feature_vec, num_words)
		feature_vecs.append(feature_vec)
	return feature_vecs

def compare(storage, feature_vecs):
	results = zeros((len(storage),len(storage)))
	min_result = 1.0
	max_result = 0.0 
	for i in range(len(storage)):
		for j in range(len(storage)):
			result = cosine_similarity(feature_vecs[i], feature_vecs[j])
			results[i][j] = result
			if result < min_result:
				min_result = result
			if result > max_result:
				max_result = result
	# 	sys.stdout.write('.') #progress
	# 	sys.stdout.flush()
	# sys.stdout.write('\n')
	#used normalize similarity scores from 0 to 1
	print 'max: ' + str(max_result)
	print 'min: ' + str(min_result)
	max_result -= min_result
	for i in range(len(results)): #normalization 
		for j in range(len(results[i])):
			results[i][j] = (results[i][j] - min_result) / max_result 
	return results