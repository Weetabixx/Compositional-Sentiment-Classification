from nltk.parse.stanford import StanfordParser  # this does not work in pycharm IDE, must be ran in a terminal
from nltk.tree import *
import sys, os, re, random, math, collections, itertools, time, pickle

PRINT_ERRORS=0

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

def bubbleSentiment(tree):
	pass
	for subtree in tree:
		if type(subtree) == Tree:  # if not a leaf

			print(("  "*level) + str(subtree.label()) + "(" )
			traverse(subtree,(level+1))
			print(("  "*level) + ")")
		else:  # if leaf node, calculate sentiment of word
			prob=pWordPos[subtree] # p(W|Positive)
			pWordNeg[subtree] # p(W|Negative)
			pWord[subtree]   # p(W) 
			print(("  "*level) + subtree)
	return(sentimentOut, combineTactic)  # sentimentOut can be -1, 0 or 1. combineTactic can be "D", "RP", "RN" or "R"


#----------------------------getInputFunctions-----------------------------------------

def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled input and splitting into lines
	posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
	posSentences = re.split(r'\n', posSentences.read())

	negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
	negSentences = re.split(r'\n', negSentences.read())

	with open('precomppolarityparsetrees.pkl', 'rb') as f:  # open precomputed parse trees
		posPolarityTrees, negPolarityTrees = pickle.load(f)


	posSentencesNokia = open('nokia-pos.txt', 'r')
	posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

	negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
	negSentencesNokia = re.split(r'\n', negSentencesNokia.read())

	with open('precompnokiaparsetrees.pkl', 'rb') as f:  # open precomputed parse trees
		posNokiaTrees, negNokiaTrees = pickle.load(f)

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
	for i in range(len(posPolarityTrees) - 1):
		if random.randint(1,10)<2:
			sentencesTest.append((posPolarityTrees[i],"positive"))
		else:
			sentencesTrain.append((posPolarityTrees[i],"positive"))

	for i in range(len(negPolarityTrees) - 1):
		if random.randint(1,10)<2:
			sentencesTest.append((negPolarityTrees[i],"negative"))
		else:
			sentencesTrain.append((negPolarityTrees[i],"negative"))

	#create Nokia Datset:
	for i in range(len(posNokiaTrees) - 1):
		sentencesNokia.append((posNokiaTrees[i],"positive"))
	for i in range(len(negNokiaTrees) - 1):
		sentencesNokia.append((negNokiaTrees[i],"negative"))

#----------------------------BayesFunctions--------------------------------------------
#TODO: remove stop words
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

	print("calculating frequencies...")
	#iterate through each sentence/sentiment pair in the training data
	#this is very slow and why parsing is precomputed probably be pre-computed and then saved to a file for use later
	#maybe the frequencies should be precomputed also for optimisation purposes
	for sentenceSentimentPair in sentencesTrain:
		sentiment = sentenceSentimentPair[1]
		sentenceTree = sentenceSentimentPair[0]
		sentenceTagged = tagtree(sentenceTree)
		taggedWords = [ wordpair[0] + "__" + wordpair[1] for wordpair in sentenceTagged]


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

		sys.stdout.write('\r' + str(allWordsTot) + " words analysed")
		sys.stdout.flush()
	print("")


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

#----------------------------runningBayes----------------------------------------------

#implement naive bayes algorithm
#INPUTS:
#  sentencesTest is a dictonary with sentences associated with sentiment 
#  dataName is a string (used only for printing output)
#  pWordPos is dictionary storing p(word|positive) for each word
#     i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#  pWordNeg is dictionary storing p(word|negative) for each word
#  pWord is dictionary storing p(word)
#  pPos is a real number containing the fraction of positive reviews in the dataset
#TODO: select sentences that are wrong and calculate P(W|Reverse-positives) and P(W|Reverse-negatives)
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord, pPos, pWordReversePos, pWordReverseNeg, pWordR):
	pNeg=1-pPos

	#variables for finding sentiment reversing words
	reverseposFeatures = [] # [] initialises a list [array]
	reversenegFeatures = [] 
	freqFalsePositive = {} # {} initialises a dictionary [hash function]
	freqFalseNegative = {}
	freqWord = {}
	dictionary = {}
	reversePosWordsTot = 0
	reverseNegWordsTot = 0
	allWordsTot = 0

	#These variables will store results (you do not need them)
	total=0
	correct=0
	totalpos=0
	totalpospred=0
	totalneg=0
	totalnegpred=0
	correctpos=0
	correctneg=0

	print("analysing frequency of sentiment reversing words")

	#for each sentence, sentiment pair in the dataset
	for sentenceSentimentPair in sentencesTest:
		sentiment = sentenceSentimentPair[1]
		sentenceTree = sentenceSentimentPair[0]
		sentenceTagged = tagtree(sentenceTree)
		taggedWords = [ wordpair[0] + "__" + wordpair[1] for wordpair in sentenceTagged]

		

		#------------------finished populating bigramList--------------
		pPosW=pPos
		pNegW=pNeg

		for word in taggedWords: #calculate over bigrams
		#        for word in wordList: #calculate over unigrams
			if word in pWord:
				if pWord[word]>0.00000001:
					pPosW *=pWordPos[word]
					pNegW *=pWordNeg[word]

		prob=0;            
		if pPosW+pNegW >0:
			prob=pPosW/float(pPosW+pNegW)


		total+=1
		if sentiment=="positive":
			totalpos+=1
			if prob>0.5:
				correct+=1
				correctpos+=1
				totalpospred+=1
			else:  # in case of false negative 
				correct+=0
				totalnegpred+=1
				if PRINT_ERRORS:
					print ("ERROR (pos classed as neg %0.2f):" %prob + taggedWords)
		else:
			totalneg+=1
			if prob<=0.5:
				correct+=1
				correctneg+=1
				totalnegpred+=1
			else:  # in case of false positive  
				correct+=0
				totalpospred+=1
				if PRINT_ERRORS:
					print ("ERROR (neg classed as pos %0.2f):" %prob + taggedWords)

		# find all sentiment reversing words
		for word in taggedWords: #calculate over bigrams
			allWordsTot += 1 # keeps count of total words in dataset
			if not (word in dictionary):
				dictionary[word] = 1
			if not (word in freqWord):  #keep count of number of times word has apeared
				freqWord[word] = 1
			else:
				freqWord[word] += 1
			if sentiment=="positive" and prob<=0.5:  # false positive
				reversePosWordsTot += 1 # keeps count of total words in false Positive class

				#keep count of each word in positive context
				if not (word in freqFalsePositive):
					freqFalsePositive[word] = 1
				else:
					freqFalsePositive[word] += 1     
			elif sentiment=="negative" and prob>0.5:  # false negative
				reverseNegWordsTot += 1 # keeps count of total words false Negative class

				#keep count of each word in positive context
				if not (word in freqFalseNegative):
					freqFalseNegative[word] = 1
				else:
					freqFalseNegative[word] += 1  

		sys.stdout.write('\r' + str(allWordsTot) + " words analysed")
		sys.stdout.flush()
	print("")

	for word in dictionary:
		#do some smoothing so that minimum count of a word is 1
		if not (word in freqFalseNegative):
			freqFalseNegative[word] = 1
		if not (word in freqFalsePositive):
			freqFalsePositive[word] = 1
		if reversePosWordsTot < 1:
			reversePosWordsTot = 1
		if reverseNegWordsTot < 1:
			reverseNegWordsTot = 1


		# Calculate P(W|Reverse-positives) 
		pWordReversePos[word] = freqFalsePositive[word] / float(reversePosWordsTot)

		# Calculate P(W|Reverse-negatives)  
		pWordReverseNeg[word] = freqFalseNegative[word] / float(reverseNegWordsTot)

		# Calculate p(word)
		pWordR[word] = freqWord[word] / float(allWordsTot) 

	acc=correct/float(total)
	print (dataName + " Accuracy (All)=%0.2f" % acc + " (%d" % correct + "/%d" % total + ")\n")

	precision_pos=correctpos/float(totalpospred)
	recall_pos=correctpos/float(totalpos)
	precision_neg=correctneg/float(totalnegpred)
	recall_neg=correctneg/float(totalneg)
	f_pos=2*precision_pos*recall_pos/(precision_pos+recall_pos);
	f_neg=2*precision_neg*recall_neg/(precision_neg+recall_neg);

	print (dataName + " Precision (Pos)=%0.2f" % precision_pos + " (%d" % correctpos + "/%d" % totalpospred + ")")
	print (dataName + " Recall (Pos)=%0.2f" % recall_pos + " (%d" % correctpos + "/%d" % totalpos + ")")
	print (dataName + " F-measure (Pos)=%0.2f" % f_pos)

	print (dataName + " Precision (Neg)=%0.2f" % precision_neg + " (%d" % correctneg + "/%d" % totalnegpred + ")")
	print (dataName + " Recall (Neg)=%0.2f" % recall_neg + " (%d" % correctneg + "/%d" % totalneg + ")")
	print (dataName + " F-measure (Neg)=%0.2f" % f_neg + "\n")


#-----------------------------Lexical aproach------------------------------------------

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
sentencesTrain=[]
sentencesTest=[]
sentencesNokia=[]

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 


#build conditional probabilities using training data
if "-precompute" in sys.argv:
	trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)
	with open('pWords.pkl', 'wb') as f:
		pickle.dump([pWordPos, pWordNeg, pWord], f)
else:
	try:
		with open('pWords.pkl', 'rb') as f:  # open precomputed parse trees
			pWordPos, pWordNeg, pWord = pickle.load(f)
	except:
		print("could not load pre-computed frequencies, exiting...")
		sys.exit()

#test bayes and find p(W|FalsePositive) p(W|Falsenegative)
pWordReversePos={}
pWordReverseNeg={}
pWordR={}


if "-pretest" in sys.argv:
	testBayes(sentencesTrain,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord, 0.5, pWordReversePos, pWordReverseNeg, pWordR)
	with open('pWordsReverse.pkl', 'wb') as f:
		pickle.dump([pWordReversePos, pWordReverseNeg, pWordR], f)
else:
	try:
		with open('pWords.pkl', 'rb') as f:  # open precomputed parse trees
			pWordReversePos, pWordReverseNeg, pWordR = pickle.load(f)
	except:
		print("could not load pre-computed frequencies of sentiment reversing words, exiting...")
		sys.exit()




#-----------------------------MAIN------------------------------------------------------



parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")  # this sets up the parser
text = input()

print("parsing the sentence...")
sample = list(parser.raw_parse(text))  # test some functions with an input

prettyprint(sample)
#traverse(sample)
tagged = tagtree(sample)
#print(tagged)
testSententce(text)


#  open the TrainingSentences  DONE
#  create parse tree of each sentence  DONE
#  for each tree:
#    if word-label pair not in lexicon:  DONE
#      add word-label pair to lexicon  DONE
#    if sentence negative:
#      increase negative counter for word-label pair  DONE
#    if sentence positive:
#      increase positive counter for word-label pair  DONE
#  test sentences, create set of wrongly identified sentences IN PROGRESS
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
