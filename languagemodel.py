from __future__ import division
import sys
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.tokenize import RegexpTokenizer
import string
import re
from collections import defaultdict
import math as math
from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
def main():
	#### GET TRAINING FILES ####
	training_files = list()
	files = sys.argv[1].split("_")
	#### GET INPUT MODEL TYPE (3S) ####
	input_model = sys.argv[3]
	gram_size,interpolate,smoothing = get_model_type(input_model) 
	##### COMPUTE THE NGRAMS FROM ALL THE FILES ####
	unigrams,bigrams,trigrams = compute_n_grams(files)
	#### READ DEV FILE. IF INTERPOLATION, INTERPOLATE ####
	dev_file = read_file(sys.argv[2])
	print "finished training lambdas....",
	if(interpolate):
		lambda_parameters = train_metaparameters(dev_file,unigrams,bigrams,trigrams)
	else:
		lambda_parameters = set_to_zero(gram_size)
	print lambda_parameters
	print "OK"
	print display_weights(lambda_parameters)
	
	#####BEGIN TESTING#####
	test(dev_file,lambda_parameters,unigrams,bigrams,trigrams,interpolate,gram_size)
	exit()

### IF NO INTERPOLATION, SET THE 
### MODEL LAMBDA TO ONE AND ALL OTHER LAMBDAS 
### TO ZERO
def set_to_zero(gram_size):
	gram_model = int(gram_size) -1
	lambdas = [0,0,0]
	lambdas[gram_model] = 1
	return lambdas

### GETS INFORMATIO ON MODEL TO BE USED
def get_model_type(user_input):
	interpolation = False
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



def display_weights(lambdas):
	print "Lambda Unigram :\t" + str(lambdas[0])
	print "Lambda Bigram : \t" + str(lambdas[1])
	print "Lambda Trigram : \t " + str(lambdas[2])



### BUILDS GRAMS FROM EACH SENTENCE, 
###AND COMPUTES THE SENTENCE PERPLEXITY,
### WRITES TO OUTPUT.TXT
def test(dev_file,lambda_parameters,unigrams,bigrams,trigrams,interpolate,gram_size):
	output = list()
	print "in test"
	dev_file.seek(0,0)
	for line_number,line in enumerate(dev_file):
		if(line_number < 2600):
			continue
		original_line = line
		original_line = original_line.replace("\n","")
		line = process(line)
		sent_tokenize_list = sent_tokenize(line)
		unigram_prob = list()
		bigram_prob = list()
		trigram_prob = list()
		for words in sent_tokenize_list:	
			tokens = word_tokenize(words)
			for index,token in enumerate(tokens):
				compute_gram_probabilities(index,words,unigrams,bigrams,trigrams,unigram_prob,bigram_prob,trigram_prob)
			perplexities = compute_gram_perplexities(unigram_prob,bigram_prob,trigram_prob,interpolate,lambda_parameters)
			gram_model = int(gram_size) -1
			write_to_stdout = str(perplexities[gram_model])
			out = str(original_line) + "\t" + str(write_to_stdout)
			output.append(out)
			
	write_to_output(output)	
	exit()
### WRITE LIST TO OUTPUT.TXT
def write_to_output(output):
	out_file = open("output.txt", 'w')
	print len(output)
	print str(output)
	for sentence in output:
		out_file.write(sentence+"\n")

def compute_gram_perplexities(unigram_prob,bigram_prob,trigram_prob,interpolate,lambdas):
	if(not interpolate):
		uni_perplexity = get_sentence_perplexity(unigram_prob)
		bi_perplexity = get_sentence_perplexity(bigram_prob)
		tri_perplexity = get_sentence_perplexity(get_interpolated_probability(unigram_prob,bigram_prob,trigram_prob,lambdas))
		return uni_perplexity,bi_perplexity,tri_perplexity

	uni_perplexity = get_sentence_perplexity(unigram_prob)
	bi_perplexity = get_sentence_perplexity(bigram_prob)
	tri_perplexity = get_sentence_perplexity(trigram_prob)
	return uni_perplexity,bi_perplexity,tri_perplexity

### INTERPOLATES UNI,BI,AND TRIGRAM PERPLEXITIES BASED ON LAMBDA 
def get_interpolated_probability(unigram_probabilities,bigram_probabilities,trigram_probabilities,lamdas):
	print "LAMBDAS : " + str(lamdas)
	inter_probablities = list()
	print trigram_probabilities 
	if(len(trigram_probabilities) < 2):
		return trigram_probabilities
	for index,probabilitiy in enumerate(trigram_probabilities):
		print probabilitiy
		if(index < 2):
			continue
		inter_probablities.append(lamdas[0]*unigram_probabilities[index-2] + lamdas[1]*bigram_probabilities[index-1] + lamdas[2]*trigram_probabilities[index])
	print "TRIGRAM PROB " + str(trigram_probabilities)
	print "INTER PROB : " + str(inter_probablities)
	return inter_probablities
	exit()

### USES A GREEDY DESCENT ALGORITHM WITHA  SPECIFIED STEPSIZE
### TO ADJUST LAMBDA VARIABLES. EXITS ON COMPLETION OF DEV OR
### OR IF ANY OF THE PARAMETERS ARE TO CLOSE TO ZERO
def train_metaparameters(dev_file,unigrams,bigrams,trigrams):
	lamda_uni = 0.10
	lamda_bi = 0.20
	lambda_tri = 0.70
	lambdas = [lamda_uni,lamda_bi,lambda_tri]
	for line_number,line in enumerate(dev_file):
		print "line num " + str(line_number)
		if(line_number >= 2600):
			return lambdas
		line = process(line)
		if(len(line) == 0):
			continue
		sent_tokenize_list = sent_tokenize(line)
		for words in sent_tokenize_list:
			tokens = word_tokenize(words)
			tokens = tokens[2:-1]
			tokens.insert(0,"<s>")
			tokens.append("</s>")
			unigram_prob = list()
			bigram_prob = list()
			trigram_prob= list()
			for i,token in enumerate(tokens):	
				
				token = is_known(token,unigrams)
				unigram_probability = get_probability(unigrams,token,None,None,index=0)
				unigram_prob.append(unigram_probability)
				
				if(i >= 1):
					bi_gram = build_gram(tokens,i,1,unigrams)
					key = (bi_gram[0],)
					val = (bi_gram[1],)
					bigram_probability = get_probability(bigrams,key,val,unigrams,index=1)
					bigram_prob.append(bigram_probability)
				if(i >= 2):
					tri_gram = build_gram(tokens,i,2,unigrams)
					tri_prefix = get_prefix(tokens,i,2,unigrams)
					tri_suffix = tri_gram[-1]
					v_size = len(unigrams)
					tri_prob = get_probability(trigrams,tri_suffix,tri_prefix,trigrams,index = 2,vocab_size = v_size)
					trigram_prob.append(tri_prob)
			uni_perplexity = get_sentence_perplexity(unigram_prob)
			bi_perplexity = get_sentence_perplexity(bigram_prob)
			tri_perplexity = get_sentence_perplexity(trigram_prob)
			perplexities = uni_perplexity,bi_perplexity,tri_perplexity
			minimum = get_min_index(perplexities)
			stepsize = 0.01
			if(not_near_zero(lambdas)):
				adjust_weights(lambdas,minimum,stepsize)


def compute_gram_probabilities(i,tokens,unigrams,bigrams,trigrams,unigram_prob,bigram_prob,trigram_prob):
	unigram_probability = get_probability(unigrams,tokens[i],None,None,index=0)
	unigram_prob.append(unigram_probability)

	if(i >= 1):
		bi_gram = build_gram(tokens,i,1,unigrams)
		key = (bi_gram[0],)
		val = (bi_gram[1],)
		bigram_probability = get_probability(bigrams,key,val,unigrams,index=1)
		bigram_prob.append(bigram_probability)

	if(i >= 2):
		tri_gram = build_gram(tokens,i,2,unigrams)
		tri_prefix = get_prefix(tokens,i,2,unigrams)
		tri_suffix = tri_gram[-1]
		v_size = len(unigrams)
		tri_prob = get_probability(trigrams,tri_suffix,tri_prefix,trigrams,index = 2,vocab_size = v_size)
		trigram_prob.append(tri_prob)

def not_near_zero(lambdas):
	for value in lambdas:
		if (value < 0.10):
			return False
	return True

def get_min_index(weights):
	return weights.index(max(weights))

### ADJUSTS 1/2 STEPSIZE TO ALL CANDIDATES NOT SELECTED
### GIVES STEPSIZE TO THE SELECTED VALUE
def adjust_weights(lambdas,minimum,stepsize):
	for x in range(len(lambdas)):
		if(x != minimum):
			lambdas[x] = lambdas[x] - stepsize
		else:
			lambdas[x] = lambdas[x] + 2*stepsize

def get_sentence_perplexity(probabilities):
	setnence_probability = 0
	for probability in probabilities:
		
		#print math.log(2,0)
		setnence_probability += math.log(probability,2)
	N = len(probabilities)
	
	per_word_entropy = (-1/N)*setnence_probability
	perplexity = (1)*math.pow(2,per_word_entropy)
	return perplexity



def print_dic(mydic):
	for key in mydic.keys():
		print "the key name is " + str(key) + " and its value is " + str(mydic[key])

def compute_n_grams(files):
	unigrams = defaultdict(int)
	bigrams = defaultdict(lambda : defaultdict(int))
	trigrams = defaultdict(lambda : defaultdict(int))
	list_of_tokens = []
	for curr_file in files:
		curr = read_file(curr_file)
		text = strip_headers(curr.read()).strip()
		sent_tokenize_list = sent_tokenize(text)
		for k,sentence in enumerate (sent_tokenize_list):
			line = preprocess(sentence)
			check_if_whitespace(sentence)
			tokens = word_tokenize(line)
			if(is_empty(tokens)):
				continue
			tokens.append("</s>")
			tokens.insert(0,"<s>")
			list_of_tokens.append(tokens)
			count(tokens,unigrams)
	vocab = build_vocab(unigrams)
	#print_dic(vocab)
	build_counts(list_of_tokens,vocab,bigrams,trigrams)

	return vocab,bigrams,trigrams



#### GETS THE PROBABILITY OF A CERTAIN EVENT.
#### IF THE EVENT IS NOT INSIDE THE MAP, ITS COUNT
#### MUST BE ZERO. IT RETURNS THE INTERPOLATD PROBABILITY
def get_probability(gram,key,value,vocab,index = 1,vocab_size=0):
	C = 0
	if(index == 0):
		return gram[is_known(key,gram)]/sum(gram.values())

	if(index == 2):
		x = (value[0],)
		y = (value[1],)
		N = vocab[x][y]
		C = gram[value][(key,)]
		V = vocab_size
	else:
		N = vocab[value[0]]
		V = len(vocab)
	
	

	if(index == 1):
		if(key in gram[value]):
			#print gram[key]

			C = gram[value][key]

	p_event = smoothing(C,V,N)
	return p_event
def smoothing(C,V,N):
	return (C + 1)/(N + V)


def generate_lookup_table(vocab,bigrams,trigrams,tokens):
	for x in range(0,len(vocab)):
		print vocab

	exit()



def build_counts(tokens,vocab,bigram,trigram):
	for token in tokens:
		count(token,None,bigrams=bigram,trigrams=trigram,vocab=vocab)

def generate_gram_distribution(vocab,seen,n,other_vocab = None):
	#distribution = defaultdict(int)
	if(other_vocab is None):
		other_vocab = vocab
	for word_one in other_vocab:
		for word_two in vocab:
			if(n == 2):
				new_word = word_one,word_two
			else:
				new_word = (word_two) + (word_one,)
			seen[new_word] = len(seen) + 1
	return seen
### RETURNS RAW COUNT OF THE NUMBER OF UNI,BI,AND TRIGRAMS IN A SENTENCE
def count(tokens,unigrams,bigrams=None,trigrams=None,vocab=None):
	b = defaultdict(lambda : defaultdict(int))
	t = defaultdict(lambda : defaultdict(int))
	for x,token in enumerate(tokens):
		if(unigrams is not None):
			unigrams[token] += 1
		if(bigrams is not None and x >= 1):
			bigram = build_gram(tokens,x,1,vocab)
			a = get_prefix(tokens,x,1,vocab)
			prev = tuple(a)
			curr = (str(is_known(tokens[x],vocab)),)
			b[a][curr] += 1
			bigrams[prev][curr] += 1
		if(trigrams is not None and x >= 2):
			trigram = build_gram(tokens,x,2,vocab)
			bigram = build_gram(tokens,x,1,vocab)
			prev = tuple(get_prefix(tokens,x,2,vocab))
			curr = (is_known(tokens[x],vocab),)
			trigrams[prev][curr] += 1

			

### RETURNS THE PREFIX KEY WHICH IS USED TO INDEX THE LOOKUP TABLE
def get_prefix(sentence,start,length,vocab):
	gram = tuple()
	for x in range(start-length,start):
		gram = gram + (is_known(sentence[x],vocab),)
	return gram

## IF THE WORD IS NOT KNOWN, RETURNS AN <UNK> TOKEN
def is_known(word,vocab):
	if (word in vocab.keys()):
		return word
	else:
		return "<unk>"


def build_vocab(unigrams):
	vocab = defaultdict(int)
	for x,word in enumerate(unigrams):
		if (unigrams[word] < 4):
			vocab["<unk>"] += 1
		else:
			vocab[word] = unigrams[word]
	return vocab 

def build_gram(sentence,start,length,vocab):
	gram = tuple()
	for x in range(start-length,start+1):
		gram = gram + (is_known(sentence[x],vocab),)
	return gram

def is_empty(tokens):
	if(len(tokens) == 0):
		return True
	return False

def preprocess(line):
	line = line.lower()
	line = line.strip()
	regex = re.compile('[%s]' % re.escape(string.punctuation))
	line = regex.sub('', line)
	return line
def process(line):
	line = line.lower()
	line = line.strip()
	line = line.replace(".","")
	line = line.replace(",","")
	return line

def skip(index):
	if(index < 272):
		return True
def check_if_whitespace(line):
	line = line.replace("\n","")
	line = line.strip()
	line = line.replace(".","")
def read_file(fname):
	print "reading file.... " + str(fname)
	input_file = open(fname, 'r')
	return input_file




main()