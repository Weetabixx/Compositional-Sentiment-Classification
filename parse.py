from nltk.parse.stanford import StanfordParser  # this does not work in pycharm IDE, must be ran in a terminal
from nltk.tree import *
import os


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
			if sub[1] == 0:  # if a leaf is returned
				sub[1] = subtree.label()
				tag = (sub[0], sub[1])
				taggedlistpairs.append(sub)
		else:  # if leaf node
			return [subtree, 0]
	return taggedlistpairs


parser=StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")  # this sets up the parser
text = input()

sample = list(parser.raw_parse(text))  # test some functions with an input
# text2 = input()
# sample += list(parser.raw_parse(text2))

prettyprint(sample)
traverse(sample)
tagged = tagtree(sample)
print(tagged)

#  open the TrainingSentences
#  create parse tree of each sentence
#  for each tree:
#    if word-label pair not in lexicon:
#      add word-label pair to lexicon
#    if sentence negative:
#      increase negative counter for word-label pair
#    if sentence positive:
#      increase positive counter for word-label pair
#  train neural net for each label?????
#  create strong general inteligence AI and get it to finish this assessment
