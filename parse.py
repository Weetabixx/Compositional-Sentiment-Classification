from nltk.parse.stanford import StanfordParser  # this does not work in pycharm IDE, must be ran in a terminal
from nltk.tree import *
import sys, os, re, random, math, collections, itertools, time, pickle


#----------------------ParseTreeFunctions------------------------------------------

def prettyprint(tree):  # takes a list of parse trees and prettyprints them
	for sub in tree:
		sub.pretty_print()

def traverse(tree, level=0):  # traverses tree and prints it out with indentation
	for subtree in tree:
		if type(subtree) == Tree:  # if not a leaf
			print(("  "*level) + str(subtree.label()) + "(" )
			traverse(subtree,(level+1))
			print(("  "*level) + ")")
		else:  # if leaf node
			print(("  "*level) + subtree)

def tagtree(tree,taggedlistpairs=[]):  # traverses tree and returns a list of tagged words(pairs)
	for subtree in tree:
		if type(subtree) == Tree:  # if not a leaf
			sub = tagtree(subtree,taggedlistpairs)
			if len(sub) >= 2:
				if sub[1] == 0:  # if a leaf is returned
					sub[1] = subtree.label()
					tag = (sub[0], sub[1])
					taggedlistpairs.append(sub)
		else:  # if leaf node
			return [subtree, 0]
	return taggedlistpairs



#----------------------------getInputFunctions-----------------------------------------

def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled input and splitting into lines
	posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
	posSentences = re.split(r'\n', posSentences.read())

	negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
	negSentences = re.split(r'\n', negSentences.read())

	posSentencesNokia = open('nokia-pos.txt', 'r')
	posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

	negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
	negSentencesNokia = re.split(r'\n', negSentencesNokia.read())

	posDictionary = open('positive-words.txt', 'r', encoding="ISO-8859-1")
	posWordList = re.findall(r"[a-z\-]+", posDictionary.read())

	negDictionary = open('negative-words.txt', 'r', encoding="ISO-8859-1")
	negWordList = re.findall(r"[a-z\-]+", negDictionary.read())

	for i in posWordList:
	    sentimentDictionary[i] = 1
	for i in negWordList:
	    sentimentDictionary[i] = -1

	#create Training and Test Datsets:
	#We want to test on sentences we haven't trained on, to see how well the model generalses to previously unseen sentences

	#create 90-10 split of training and test data from movie reviews, with sentiment labels    
	for i in posSentences:
	    if random.randint(1,10)<2:
	        sentencesTest[i]="positive"
	    else:
	        sentencesTrain[i]="positive"

	for i in negSentences:
	    if random.randint(1,10)<2:
	        sentencesTest[i]="negative"
	    else:
	        sentencesTrain[i]="negative"

	#create Nokia Datset:
	for i in posSentencesNokia:
	        sentencesNokia[i]="positive"
	for i in negSentencesNokia:
	        sentencesNokia[i]="negative"

#----------------------------BayesFunctions--------------------------------------------

#TODO: need to change bayes to use parse tree and implement a re-train function
#calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
#TODO: W --> Word__PartOfSpeechTag
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
	posFeatures = [] # [] initialises a list [array]
	negFeatures = [] 
	freqPositive = {} # {} initialises a dictionary [hash function]
	freqNegative = {}
	dictionary = {}
	posWordsTot = 0
	negWordsTot = 0
	allWordsTot = 0
	# initialises parser that creates parse trees 
	stanParser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

	print("parsing training data ... ")
	parsedSentences = 0
	totSentences = len(sentencesTrain.items())
	print(str(totSentences)+" sentences to parse... ")
	startTime = time.time()

	#iterate through each sentence/sentiment pair in the training data
	#this is very slow and should probably be pre-computed and then saved to a file for use later
	for sentence, sentiment in sentencesTrain.items():
		try: # try statment due to some sentences not being parsed properly
			#print(type(sentence))
			wordList = re.findall(r"[\w']+", sentence)
			stringSentence = ""
			for x in range(len(wordList) - 1):
				stringSentence += str(wordList[x]) + " "
				#print(str(wordList[x]))
				#print(type(wordList[x]))
			sentenceTree = list(stanParser.raw_parse(stringSentence))
			#print(sentenceTree)
			sentenceTagged = tagtree(sentenceTree)
			taggedWords = [ wordpair[0] + "__" + wordpair[1] for wordpair in sentenceTagged]
			parsedSentences += 1
			i = str(parsedSentences) + "/" + str(totSentences) + "---" + str(parsedSentences/totSentences * 100) + "%"
			sys.stdout.write("\rSentences parsed: %i" % i)
			sys.stdout.flush()

			#TO DO:
			#Populate bigramList (initialised below) by concatenating adjacent words in the sentence.
			#You might want to seperate the words by _ for readability, so bigrams such as:
			#You_might, might_want, want_to, to_seperate.... 

			bigramList=wordList #initialise bigramList
			for x in range(len(wordList)-1):
				bigramList.append(wordList[x]+"_" + wordList[x+1])


			#-------------Finish populating bigramList ------------------#

			#TO DO: when you have populated bigramList, uncomment out the line below and , and comment out the unigram line to make use of bigramList rather than wordList

			for word in taggedWords: #calculate over bigrams
				# for word in wordList: #calculate over unigrams
				allWordsTot += 1 # keeps count of total words in dataset
				if not (word in dictionary):
					dictionary[word] = 1
				if sentiment=="positive" :
					posWordsTot += 1 # keeps count of total words in positive class

					#keep count of each word in positive context
					if not (word in freqPositive):
					    freqPositive[word] = 1
					else:
					    freqPositive[word] += 1    
				else:
					negWordsTot+=1# keeps count of total words in negative class

					#keep count of each word in positive context
					if not (word in freqNegative):
					    freqNegative[word] = 1
					else:
					    freqNegative[word] += 1
		except:
			print("-could not parse sentence-")
			totSentences -= 1
	finishTime = time.time()
	tottime = startTime - finishTime
	print("took " + str(tottime) + " seconds to calculate parse trees for training data")


	for word in dictionary:
		#do some smoothing so that minimum count of a word is 1
		if not (word in freqNegative):
			freqNegative[word] = 1
		if not (word in freqPositive):
			freqPositive[word] = 1

		# Calculate p(word|positive)
		pWordPos[word] = freqPositive[word] / float(posWordsTot)

		# Calculate p(word|negative) 
		pWordNeg[word] = freqNegative[word] / float(negWordsTot)

		# Calculate p(word)
		pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 
	# Saving the objects:
	with open('objs.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
		pickle.dump([pWordPos, pWordNeg, pWord], f)

#----------------------------runningBayes----------------------------------------------

def testSententce(sentence):
	Words = re.findall(r"[\w']+", sentence)
	score=0
	for word in Words:
		if word in sentimentDictionary:
			score+=sentimentDictionary[word]
			print(word + " is " + str(sentimentDictionary[word]))
	print(score)

#----------------------------VariableInitialisations-----------------------------------

sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

# check if we need to precompute the probabilities of positive and negative and total words
if len(sys.argv) > 1:
	if sys.argv[1] == "-precompute":
		#build conditional probabilities using training data
		trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)
	else:
		print("these arguments don't make sense, use is:")
		print("python3 parse.py -precompute")
		print("where '-precompute' is optional")
		sys.exit()
else: #or load precomputed probabilities
	# Getting back the objects:
	with open('objs.pkl') as f:  # Python 3: open(..., 'rb')
		pWordPos, pWordNeg, pWord = pickle.load(f)


#-----------------------------MAIN------------------------------------------------------



parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")  # this sets up the parser
text = input()

print("parsing the sentence...")
sample = list(parser.raw_parse(text))  # test some functions with an input
# text2 = input()
# sample += list(parser.raw_parse(text2))

prettyprint(sample)
#traverse(sample)
tagged = tagtree(sample)
print(tagged)
testSententce(text)


#  open the TrainingSentences
#  create parse tree of each sentence
#  for each tree:
#    if word-label pair not in lexicon:
#      add word-label pair to lexicon
#    if sentence negative:
#      increase negative counter for word-label pair
#    if sentence positive:
#      increase positive counter for word-label pair
#  test sentences, create set of wrongly identified sentences
#  for each tree in sentence:
#    if sentence is false positive:
#      increase reverse positive count for word-label pair
#    if sentence is false negative:
#      increase reverse negative count for word-label pair
#  calculate which words pass threshhold for being labeled as reverse|reverse-negative|reverse-positive
#
#
#
#
#  compare to traditional bayes
#  write rules on how nodes connect their sentiment 
#
#
#  for each tree if 
