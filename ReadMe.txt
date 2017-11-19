installing prerequisites(assuming python3 is installed):---------------------------------------------------------------------
sudo pip install -U nltk

wget http://nlp.stanford.edu/software/stanford-ner-2015-04-20.zip
wget http://nlp.stanford.edu/software/stanford-postagger-full-2015-04-20.zip
wget http://nlp.stanford.edu/software/stanford-parser-full-2015-04-20.zip
# Extract the zip file.
unzip stanford-ner-2015-04-20.zip
unzip stanford-parser-full-2015-04-20.zip
unzip stanford-postagger-full-2015-04-20.zip

# replace /home/path/to/stanford/tools/ with where the files are downloaded/installed
export STANFORDTOOLSDIR=/home/path/to/stanford/tools/

export CLASSPATH=$STANFORDTOOLSDIR/stanford-postagger-full-2015-04-20/stanford-postagger.jar:$STANFORDTOOLSDIR/stanford-ner-2015-04-20/stanford-ner.jar:$STANFORDTOOLSDIR/stanford-parser-full-2015-04-20/stanford-parser.jar:$STANFORDTOOLSDIR/stanford-parser-full-2015-04-20/stanford-parser-3.5.2-models.jar

export STANFORD_MODELS=$STANFORDTOOLSDIR/stanford-postagger-full-2015-04-20/models:$STANFORDTOOLSDIR/stanford-ner-2015-04-20/classifiers


the instructions for these instalations were taken from stackoverflow, for more details see https://stackoverflow.com/questions/13883277/stanford-parser-and-nltk from the second answer as the first does no longer work

running:------------------------------------------------------------------------------------

python3 parse.py -precomp -pretest
this will run the system, -precomp and -pretest are optional and should only be included in the first run of the system, or after a large update as these take some additional time, without these included the run time should be much shorter and probably less than 5 minutes

errors:-------------------------------------------------------------------------------------

if there is an error: NLTK was unable to find stanford...
	then the export enviroment variables should be added to the .bashrc file

if there is an error about not being able to find an .pkl file, this means that the precomputations have not been run. To fix this run python3 precomp.py and depending on which .pkl file is missing the code in precomp.py may have to be modified to precompute the parse trees for the specific training data.
if using a new data set precomp.py will have to be modified and ran, this may take a while(from 20 minutes to a day depending on the size of the data set)


other:------------------------------------------------------------------------

if you are interested in what the tags mean, this explains all the tags used: http://web.mit.edu/6.863/www/PennTreebankTags.html
