from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample=gutenberg.raw("bible-kjv.txt")
sen=sent_tokenize(sample)

for i in sen:
    print(i)
