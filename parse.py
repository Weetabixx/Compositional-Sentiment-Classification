from nltk.parse.stanford import StanfordParser  # this does not work in pycharm IDE, must be ran in a terminal
from nltk.tree import *
import os
parser=StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
sample = list(parser.raw_parse("the quick brown fox jumps over the lazy dog"))


def prettyprint(tree):  # takes a list of parse trees and prettyprints them
	for sub in tree:
		sub.pretty_print()

prettyprint(sample)
