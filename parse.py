from nltk.parse.stanford import StanfordParser
parser=StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")
print(list(parser.raw_parse("the quick brown fox jumps over the lazy dog")))
