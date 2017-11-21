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
	childSentimentTactics = []
	for subtree in tree:
		if type(subtree) == Tree:  # if not a leaf
			sentimenttacticPair = bubbleSentiment(subtree)
			childSentimentTactics.append(sentimenttacticPair)
		else:  # if leaf node, return leaf and sentiment tactic pair
			tag = tree.label()
			taggedW = subtree + "__" + tag
			#print("leaf is:" + taggedW)
			calculatedPair = lexiSentiment(taggedW)
			childSentimentTactics.append(calculatedPair)
	if len(childSentimentTactics) == 1: # if there was only one child just bubble that up
		#print("only one child, " + str(childSentimentTactics))
		return childSentimentTactics[0]
	#  calculate overall sentiment of child nodes
	#  check if there is a sentiment reversing tactic
	#print("combining: " + str(childSentimentTactics))
	reversePositives = False
	reverseNegatives = False
	inMixed = False
	combineTactic = "D"
	for tacticSentimentPair in childSentimentTactics:
		if tacticSentimentPair[1] == "R":
			reversePositives = True
			reverseNegatives = True
		elif tacticSentimentPair[1] == "RP":
			reversePositives = True
		elif tacticSentimentPair[1] == "RN":
			reverseNegatives = True
		if tacticSentimentPair[1] == "M":
			inMixed = True
			combineTactic = "M"
	#  add up all of the sentiments
	sentimentOut = 0
	mixed = False
	for tacticSentimentPair in childSentimentTactics:
		sentimentOut += tacticSentimentPair[0]
		if sentimentOut != 0:
			mixed = True
	#  check combine tactic and adjust sentiment accordingly
	if reverseNegatives and reversePositives:  # reverse sentiment 
		sentimentOut *= -1
	elif reverseNegatives:  # reverse negatives
		sentimentOut = sentimentOut*sentimentOut
	elif reversePositives:  # reverse positives
		sentimentOut = sentimentOut*sentimentOut* -1
	#print("combined sentiment: " + str(sentimentOut))
	
	#  normalise output to either -1, 1 or 0
	if sentimentOut >= 1:
		sentimentOut = 1
	elif sentimentOut <= -1:
		sentimentOut = -1
	elif sentimentOut == 0 and mixed:  # if there was a mixed sentiment just pick one instead of neutral sentiment
		combineTactic = "M"
		if inMixed:  # if some of the previous inputs where mixed, ignore them in this calculation
			sentimentOut = 0
			mixed = False
			for tacticSentimentPair in childSentimentTactics:
				if tacticSentimentPair[1] != "M":  # ignore mixed inputs
					sentimentOut += tacticSentimentPair[0]
					if sentimentOut != 0:
						mixed = True
			#  check combine tactic and adjust sentiment accordingly
			if reverseNegatives and reversePositives:  # reverse sentiment 
				sentimentOut *= -1
			elif reverseNegatives:  # reverse negatives
				sentimentOut = sentimentOut*sentimentOut
			elif reversePositives:  # reverse positives
				sentimentOut = sentimentOut*sentimentOut* -1

			#  normalise output to either -1, 1 or 0
			if sentimentOut >= 1:
				sentimentOut = 1
			elif sentimentOut <= -1:
				sentimentOut = -1
			elif sentimentOut == 0 and mixed:  # if there was a mixed sentiment just pick one instead of neutral sentiment
				sentimentOut = childSentimentTactics[-1][0]
		else:
			sentimentOut = childSentimentTactics[-1][0]
	
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

	posDictionary = open('opinion-lexicon-English/positive-words.txt', 'r', encoding="ISO-8859-1")
	posWordList = re.findall(r"[a-z\-]+", posDictionary.read())

	negDictionary = open('opinion-lexicon-English/negative-words.txt', 'r', encoding="ISO-8859-1")
	negWordList = re.findall(r"[a-z\-]+", negDictionary.read())

	revDictionary = open('reverse.txt', 'r', encoding="ISO-8859-1")
	revWordList = re.findall(r"[a-z\-']+", revDictionary.read())

	revPosDictionary = open('reversePos.txt', 'r', encoding="ISO-8859-1")
	revPosWordList = re.findall(r"[a-z\-']+", revPosDictionary.read())

	revNegDictionary = open('reverseNeg.txt', 'r', encoding="ISO-8859-1")
	revNegWordList = re.findall(r"[a-z\-']+", revNegDictionary.read())

	for i in posWordList:
	    sentimentDictionary[i] = 1
	for i in negWordList:
	    sentimentDictionary[i] = -1

	for i in revWordList:
		tagDictionary[i] = "R"
	for i in revPosWordList:
		tagDictionary[i] = "RP"
	for i in revNegWordList:
		tagDictionary[i] = "RN"

	#create Training and Test Datsets:
	#We want to test on sentences we haven't trained on, to see how well the model generalses to previously unseen sentences

	#create 90-10 split of training and test data from movie reviews, with sentiment labels    
	for i in range(len(posPolarityTrees) - 1):
		#if random.randint(1,10)<2:
		if True:
			sentencesTest.append((posPolarityTrees[i],"positive"))
		else:
			sentencesTrain.append((posPolarityTrees[i],"positive"))

	for i in range(len(negPolarityTrees) - 1):
		#if random.randint(1,10)<2:
		if True:
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
def testReverseBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord, pPos, pWordReversePos, pWordReverseNeg, pWordR):
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


#uses the frequencies calculated in the test and training of bayes to decide the sentiment and combine tactic of a given tagged word
def calculateSentiment(taggedWord):
	sentiment = 0  #defaults
	tactic = "D"

	# calculate sentiment based on bayes approach
	if taggedWord in pWord:
		probPos = (pWordPos[taggedWord] * 0.5) / pWord[taggedWord]
		probNeg = (pWordNeg[taggedWord] * 0.5) / pWord[taggedWord]
		if probPos > 0.5:
			sentiment = 1
		elif probNeg > 0.5:
			sentiment = -1

	#calculate combine tactic based on bayes
	if taggedWord in pWordR:
		probRevPos = (pWordReversePos[taggedWord] * 0.5) / pWordR[taggedWord]
		probRevNeg = (pWordReverseNeg[taggedWord] * 0.5) / pWordR[taggedWord]
		if probRevPos > 0.5:
			tactic = "RP"
		if probRevNeg > 0.5:
			tactic = "RN"
		if probRevNeg > 0.5 and probRevPos > 0.5:
			tactic = "R"
	
	# could replace wirh lexical approach for different results
	return(sentiment, tactic)


#-----------------------------Lexical aproach------------------------------------------

def lexiSentiment(taggedWord):  # returns pair of -1, 0 or 1 and "D" or "R" for sentiment reversers
	word = taggedWord.split("__")[0]
	if word in sentimentDictionary:
		score = sentimentDictionary[word]
	else:
		score = 0

	tag = "D"
	if word in tagDictionary:
		tag = tagDictionary[word]

	return(score, tag)


def testSententce(sentence):
	Words = re.findall(r"[\w']+", sentence)
	score=0
	for word in Words:
		if word in sentimentDictionary:
			score+=sentimentDictionary[word]
			print(word + " is " + str(sentimentDictionary[word]))
	print(score)
	if score > 0 :
		return "positive"
	if score < 0 :
		return "negative"
	if score == 0 :
		return "neutral"

def testLexicalCompositionalApproach(testSet, dataName):  # test the efficiency of a lexical approach given a set of sentence-sentiment pairs
	total=0
	correct=0
	totalpos=0
	totalpospred=0
	totalneg=0
	totalnegpred=0
	correctpos=0
	correctneg=0

	print("testing sentences...")
	for sentenceSentimentPair in testSet:
		total += 1
		actualSentiment = sentenceSentimentPair[1]
		sentence = sentenceSentimentPair[0]
		sentiment = bubbleSentiment(sentence)  # does the actual calculation
		sentiment = sentiment[0]
		if actualSentiment == "positive":
			totalpos += 1
			if sentiment == 1:
				correct += 1
				correctpos += 1
				totalpospred += 1
			elif sentiment == -1:
				totalnegpred += 1
		if actualSentiment == "negative":
			totalneg += 1
			if sentiment == -1:
				correct += 1
				correctneg += 1
				totalnegpred += 1
			elif sentiment == 1:
				totalpospred += 1
		sys.stdout.write('\r' + str(total) + " sentences analysed")
		sys.stdout.flush()
	print("")

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






#----------------------------VariableInitialisations-----------------------------------

sentimentDictionary={} # {} initialises a dictionary [hash function]
tagDictionary={}
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
	testReverseBayes(sentencesTrain,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord, 0.5, pWordReversePos, pWordReverseNeg, pWordR)
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

testLexicalCompositionalApproach(sentencesNokia, "Nokia (Test Data, Compositional-Based)\t")
testLexicalCompositionalApproach(sentencesTest, "Films (Test Data, Compositional-Based)\t")

parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")  # this sets up the parser
text = input("try a sentence yourself:")

print("parsing the sentence...")
sample = list(parser.raw_parse(text))  # test some functions with an input

prettyprint(sample)
# traverse(sample)
# tagged = tagtree(sample)
# print(tagged)
testSententce(text)
bubbleResult = bubbleSentiment(sample)
print("bubble result:")
print(bubbleResult)