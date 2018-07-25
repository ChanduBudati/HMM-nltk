# HMM-nltk
simple hmm implementation for pos tagging in python


• The program prompts the user for the Brown corpus files to be used as training data. These must
be from the news category and should be input in a single line, separated by a space.
Example: ca03 ca05 ca18 ca30
• The program should also prompt the user for a single news category file to be used as test data.
Example: ca27
• Training – To determine the tag transition probabilities and word/tag probabilities, process the files by
sentences (using the corpus sents() method). Each training sentence should be tagged using the NLTK default
tagger (accessed through the pos_tag() method). Tag bigrams should be extracted using the NLTK bigrams()
method. Note that you will need to add a start sentence tag. At the end of training, the transition
probabilities and observation likelihood matrices should be in place.
• Testing – Tag each test sentence using your HMM tagger.
• Evaluation – Evaluate the accuracy of your tagger by comparing its tags to those of the default tagger. Note that
the default tagger does not always correctly tag words. Assume those tags are correct in this case.
• Output – The program should display the following output:
• The number of correct and incorrect tags, along with a percent accuracy.
