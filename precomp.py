from nltk.parse.stanford import StanfordParser  # this does not work in pycharm IDE, must be ran in a terminal
from nltk.tree import *
import pickle

posSentencesNokia = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
negSentencesNokia = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
#initialises parser that creates parse trees 
stanParser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

posSentencesNokia = posSentencesNokia.readlines()
posSentencesNokia = [x.strip() for x in posSentencesNokia]

negSentencesNokia = negSentencesNokia.readlines()
negSentencesNokia = [x.strip() for x in negSentencesNokia]

posTrees = []

totSentences = len(negSentencesNokia) + len(posSentencesNokia)
parsedSentences = 0

for sentence in posSentencesNokia:
	parsedSentences += 1
	try:
		sentenceTree = list(stanParser.raw_parse(sentence))
		posTrees.append(sentenceTree)
	except:
		print("--failed parse--")
	print(str(parsedSentences) + "/" + str(totSentences))

negTrees = []

for sentence in negSentencesNokia:
	parsedSentences += 1
	try:
		sentenceTree = list(stanParser.raw_parse(sentence))
		negTrees.append(sentenceTree)
	except:
		print("--failed parse--")
	print(str(parsedSentences) + "/" + str(totSentences))

# Saving the objects:
with open('precomppolarityparsetrees.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
	pickle.dump([posTrees, negTrees], f)