from nltk.parse.stanford import StanfordParser  # this does not work in pycharm IDE, must be ran in a terminal
from nltk.tree import *
import os


def prettyprint(tree):  # takes a list of parse trees and prettyprints them
	for sub in tree:
		sub.pretty_print()

def traverse(tree, level=0):
	for subtree in tree:
		if type(subtree) == Tree:  # if not a leaf
			print(("  "*level) + str(subtree.label()) + "(" )
			traverse(subtree,(level+1))
			print(("  "*level) + ")")
		else:  # if leaf node
			print(("  "*level) + subtree)


parser=StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
text = input()

sample = list(parser.raw_parse(text))
# text2 = input()
# sample += list(parser.raw_parse(text2))

prettyprint(sample)
traverse(sample)
