# -*- coding: utf-8 -*-
from __future__ import division
import sys
from collections import defaultdict
import pprint
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.tokenize import RegexpTokenizer
import pickle
import math as math
def main():
	use_model = sys.argv[1]
	gram_size,interpolate,smoothing = get_model(use_model)
	fname = sys.argv[2]
	input_file = read_file(fname)
	##### train ######
	unigrams,bigrams,trigrams = n_grams(input_file)
	vocab = unigrams
	#probability = compute_probabilities(unigrams,bigrams,trigrams)
		
	if(smoothing):
		oov = get_unks(unigrams,bigrams,trigrams)
		replace_oov_with_unks(oov,input_file)
		counts = count_with_unks()
		gram_probabilities,lookup = perform_smoothing(counts[0],counts[1],counts[2])
		unigram_probabilities = gram_probabilities[0]
		bigram_probabilities = gram_probabilities[1]
		trigram_probabilities = gram_probabilities[2]
	else:
		gram_probabilities,lookup = compute_probabilities(unigrams,bigrams,trigrams)
		
	#### end train #####
	####  dev #####
	dev_name = sys.argv[3]
	dev_file = read_file(dev_name)
	#print dev_file.read()
	lambda_unigram = 0.0
	lambda_bigram = 0.0
	lambda_trigram = 0.0
	if(gram_size == "3"):
		lambda_unigram = 0.10
		lambda_bigram = 0.40
		lambda_trigram = 0.50
	elif(gram_size == "2"):
		lambda_bigram = 0.75
		lambda_unigram = 0.25 
	if(interpolate):
		lambdas = train_lambda_values(dev_file,lambda_unigram,lambda_bigram,lambda_trigram,gram_probabilities,lookup,vocab,gram_size)
		test(unigram_probabilities,bigram_probabilities,trigram_probabilities,lambdas,vocab)
	else:
		probability_to_use = gram_probabilities[int(gram_size)-1]
		test_no_interpolation(probability_to_use,vocab,int(gram_size))


	################################### TESTING ########################################################
	


def get_model(user_input):
	interpolation = None
	gram = None
	gram = user_input[0]
	smoothing = False
	if(user_input == "1"):
		smoothing = True
	user_interpolatation = None
	if(len(user_input) > 1):
		user_interpolatation = user_input[1]
	if(user_interpolatation == "s"):
		interpolation = True
		smoothing = True

	return gram,interpolation,smoothing


def test_no_interpolation(gram,vocab,gram_size):
	test_fname = sys.argv[4]
	test_file = read_file(test_fname)
	probabilities = list()
	for line in test_file:
		line = remove_punctuation(line)
		line = line.replace("\n","")
		sentence = line.split()
		if(len(sentence) == 0):
			continue
		for x in range (gram_size-1,len(sentence)):
			if(gram_size > 1):
				curr_gram = build_gram(sentence,x,gram_size-1,vocab)
			else:
				curr_gram = is_known(sentence[x],gram)	
			probability = gram[curr_gram]
			probabilities.append(probability)
		perplexity = get_sentence_perplexity(probabilities)
		probabilities = []
		print line.replace("\n","") + " : " + str(perplexity)





def is_known(word,vocab):
	if (word in vocab.keys()):
		return word
	else:
		return "<unk>"




#### CALCULATES THE PER WORD PERPLEXITY FOR EACH SENTENCE.
#### THE FORMULA IT USES IS GIVEN IN THE JURAFSKY AND MARTIN BOOK (2^(PER_WORD_ENTROPY))
def test(unigram_probabilities,bigram_probabilities,trigram_probabilities,lambdas,vocab):
	test_fname = sys.argv[4]
	test_file = read_file(test_fname)
	probabilities = list()
	for line in test_file:
		line = remove_punctuation(line)
		line = line.replace("\n","")
		sentence = line.split()
		if(len(sentence) == 0):
			continue
		for x in range (2,len(sentence)):
			trigram = build_gram(sentence,x,2,vocab)
			bigram = build_gram(sentence,x,1,vocab)
			unigram = sentence[x]
			#bigram = build
			str(trigram_probabilities[trigram])
			unigram_probability = unigram_probabilities[unigram]
			bigram_probability = bigram_probabilities[bigram]
			trigram_probability = trigram_probabilities[trigram]
			unigram_probabilities[unigram]
			probability = get_interpolated_probability(unigram_probability,bigram_probability,lambdas,trigram_probability)
			probabilities.append(probability)
		
		
	
		perplexity = get_sentence_perplexity(probabilities)
		print line.replace("\n","") + " : " + str(perplexity)

def get_sentence_perplexity(probabilities):
	setnence_probability = 0
	#print probabilities
	for probability in probabilities:
		if(probability == 0):
			return float("inf")
		setnence_probability += math.log(probability,2)
	N = len(probabilities)
	per_word_entropy = (-1/N)*setnence_probability
	perplexity = math.pow(2,per_word_entropy)
	return perplexity


def get_interpolated_probability(unigram_probabilities,bigram_probabilities,lamdas,trigram_probabilities=None):
	if(len(lamdas) == 2):
		return lamdas[0]*unigram_probabilities + lamdas[1]*bigram_probabilities	
	return lamdas[0]*unigram_probabilities + lamdas[1]*bigram_probabilities + lamdas[2]*trigram_probabilities

def build_gram(sentence,start,length,vocab):
	gram = tuple()
	for x in range(start-length,start+1):
		gram = gram + (is_known(sentence[x],vocab),)
	return gram
def print_dic(mydic):
	for key in mydic.keys():
		print "the key name is " + str(key) + " and its value is " + str(mydic[key])

	#### end dev #####
def replace_oov_with_unks(oov,input_file):
	target = open("unk.txt", 'w')
	sentences_with_unks = list()
	input_file.seek(0,0)
	for line in input_file:
		tokenizer = RegexpTokenizer(r'\w+')
		sentence = tokenizer.tokenize(line)
		sentence.append("</s>")
		sentence.insert(0,"<s>")
		for x in range(0,len(sentence)):
			if(sentence[x] in oov):
				sentence[x] = "<unk>"
		sentences_with_unks.append(sentence)
	pickle.dump(sentences_with_unks, target)
## GENERTATES TABLE OF ALL POSSIBLE GRAMS
def generate_gram_distribution(vocab,seen,n,other_vocab = None):
	distribution = defaultdict(int)
	if(other_vocab is None):
		other_vocab = vocab
	for word_one in other_vocab:
		for word_two in vocab:
			if(n == 2):
				new_word = word_one,word_two
			else:
				new_word = (word_two) + (word_one,)
			seen[new_word] += 0
	return seen

def get_vocab_tuple(vocab):
	vocab_list = list()
	for word in vocab:
		vocab_list.append((word))
	return vocab_list	

# COUNTS ALL THE UNIGRAM, BIGRAM,TRIGRAM WITH <UNK> PARAMETER
def count_with_unks():
	target = open("unk.txt",'r')
	sentences = pickle.load(target)
	counts = compute_n_grams(sentences)
	unigram_counts = counts[0]
	bigram_vocab = counts[0].keys()
	seen_bigrams = counts[1]
	bigram_counts = generate_gram_distribution(bigram_vocab,seen_bigrams,2)
	seen_trigrams = counts[2]
	trigram_vocab = bigram_counts.keys()
	trigram_counts = generate_gram_distribution(trigram_vocab,seen_trigrams,3,bigram_vocab)
	return unigram_counts, bigram_counts,trigram_counts

def perform_smoothing(unigrams,bigrams,trigrams):
	uni_prob = defaultdict(int)
	bi_prob = defaultdict(int)
	tri_prob = defaultdict(int)
	grams = unigrams,bigrams,trigrams
	prob = uni_prob,bi_prob,tri_prob
	new_counts = list()
	for x in range(0,len(grams)): 
		smooth(grams[x])
		


	return compute_probabilities(unigrams,bigrams,trigrams)

def smooth(gram):
	for word in gram:
		count = gram[word]
		#gram[word] = count + 1
	

def get_unks(unigrams,bigrams,trigrams):
	oov = set()
	for word in unigrams:
		if(unigrams[word] < 2):
			oov.add(word)
	return oov
def is_known(word,vocab):
	if word not in vocab:
		return "<unk>"
	else:
		return word


def not_near_zero(lambdas):
	for value in lambdas:
		if (value < 0.10):
			return False
	return True
## A GREEDY DESCENT ALGORITHM
## THAT REDISTRIBUTES WEIGHT FROM INCORRECTLY CHOSEN
## GRAMS, TO GRAMS THAT CHOOSE CORRECTLY. RUNS UNTIL END OF FILE
## OR IF ONE VALUE GETS TO CLOSE TO ZERO.
def train_lambda_values(dev_file,lambda_unigram,lambda_bigram,lambda_trigram,probability,lookup,vocab,gram_size):
	start = 0
	end = 3
	unigram_probabilities = probability[0]
	bigram_probabilities = probability[1]
	trigram_probabilities = probability[2]
	if(gram_size == "3"):
		lambdas = [lambda_unigram,lambda_bigram,lambda_trigram]
	else:
		lambdas = [lambda_unigram,lambda_bigram]
	print lambdas
	dev_file.seek(0,0)
	for line in dev_file.readlines():
		words = line.split(" ")	
		for x in range(0,len(words)):
			curr_word = is_known(words[x],vocab)
			#exit()
			if(x < 2):
				continue
			prev_word = is_known(words[x-1],vocab)
			prev_prev_word = is_known(words[x-2],vocab)
			unigram_pro = unigram_probabilities[curr_word]
			bigram_pro = get_assigned_probability(prev_word,lookup,curr_word)
			trigram_pro = get_assigned_probability((prev_prev_word,prev_word),lookup,curr_word)
			weights = None
			if(gram_size == "2"):
				weights = unigram_pro,bigram_pro
			else:
				weights = unigram_pro,bigram_pro,trigram_pro
			minimum = get_min_index(weights)
			if(not_near_zero(lambdas)):
				adjust_weights(lambdas,minimum,0.01)
			else:
				return lambdas

			uni_prob = unigram_probabilities[curr_word]
			bi_prob = bigram_probabilities[prev_word]
	
	print "The Adjusted Lamda Values Are : \t"  + str(lambdas)
	return lambdas

def adjust_weights(lambdas,minimum,stepsize):
	other_grams = len(lambdas) - 1
	for x in range(0,len(lambdas)):
		if(x != minimum):
			lambdas[x] = lambdas[x] - stepsize
		else:
			lambdas[x] = lambdas[x] + (other_grams)*stepsize


def get_min_index(weights):
	return weights.index(max(weights))

def get_assigned_probability(prev,lookup,actual):
	candidates_weights = lookup[prev]
	candidates = [ seq[0] for seq in candidates_weights ]
	weights = [ seq[1] for seq in candidates_weights ]
	acIndex = candidates.index(actual)
	return weights[acIndex]

def weighted_pick(prev,lookup):
	candidates_weights = lookup[prev]
	candidates = [ seq[0] for seq in candidates_weights ]
	weights = [ seq[1] for seq in candidates_weights ]
	bigram_pick = choice(candidates, 1, weights) 
	return bigram_pick

def compute_probabilities(unigrams,bigrams,trigrams):
	grams = unigrams,bigrams,trigrams
	probability = defaultdict(int)
	index = 0
	uni_prob = defaultdict(int)
	bi_prob = defaultdict(int)
	tri_prob = defaultdict(int)
	lookup = defaultdict(list)
	prob = uni_prob,bi_prob,tri_prob
	type_gram = 0
	for gram in grams:
		if(index == 0):
			no_condition(gram,prob[type_gram])
		else:
			condition(grams,prob[type_gram],index,lookup,"ADD_ONE")
		index = index + 1
		type_gram += 1
	pp = pprint.PrettyPrinter(indent=2)
	#pp.pprint(lookup)
	#exit()
	return prob,lookup
		
def condition(grams,probability,index,lookup,smoothing=None):
	v = len(grams[0])
	
	prev_gram = grams[index-1]
	curr_gram = grams[index]
	for word in curr_gram:
		prev_word = ""
		if(index == 1):
			prev_word = word[0]
		else:
			prev_word = word[0:index]

			
		prev_count = prev_gram[prev_word]
		curr_count = curr_gram[word]

		if(smoothing == None):
			conditoned_probability = calc_conditional_probability(prev_count,curr_count)
		elif(smoothing == "ADD_ONE"):
			conditoned_probability = calc_conditional_probability(prev_count + v ,curr_count+1)
		probability[word] = conditoned_probability
		lookup[prev_word].append((word[-1],conditoned_probability))
	
	## GET PROBABABILITY CONDITIONED ON (x+1 gram)/(x gram)
def calc_conditional_probability(prev_count,curr_count):
	return curr_count/prev_count

## CALCULATE RAW PROBABILITY WITH NO INFORMATION
def no_condition(gram,probability):
	total_count = sum(gram.values())
	for words in gram:
		count = gram[words]
		probability[words] = count/total_count

def read_file(fname):
	print "reading file.... " + fname
	input_file = open(fname, 'r')
	return input_file

# RETURNS ALL UNI,BI,AND TRIGRAMS FROM A GIVEN SENTENCE
def compute_n_grams(sentences):
	unigrams = defaultdict(int)
	bigrams = defaultdict(int)
	trigrams = defaultdict(int)
	grams = unigrams,bigrams,trigrams
	for words in sentences:
		count_words(words,grams)
	return unigrams,bigrams,trigrams
#RETURNS ALL UNI,BI,AND TRI GRAMS IN A FILE
def n_grams(input_file):
	unigrams = defaultdict(int)
	bigrams = defaultdict(int)
	trigrams = defaultdict(int)
	grams = unigrams,bigrams,trigrams
	for line in input_file:
		tokenizer = RegexpTokenizer(r'\w+')
		sentences = tokenizer.tokenize(line)
		#sentences = sent_tokenize(line)
		sentences.append("</s>")
		sentences.insert(0,"<s>")
		count_words(sentences,grams)
	return unigrams,bigrams,trigrams

def remove_punctuation(line):
	return line.replace(",","").replace(".","")	
## COUNTS OCCURANCE OF WORDS IN A LINE
def count_words(line,grams):
	unigrams = grams[0]
	bigrams = grams[1]
	trigrams = grams[2]
	words = line
	index = 0
	for index in range(0,len(words)):
		curr_word = words[index]
		biword = ""
		triword = ""
		unigrams[(curr_word)] += 1
		if(index >= 1):
			prev_word = words[index-1]
			biword = prev_word,curr_word
			bigrams[biword] += 1
		if(index >= 2):
			prev_prev_word = words[index-2]
			triword = (prev_prev_word,) + biword
			trigrams[triword] += 1




main()