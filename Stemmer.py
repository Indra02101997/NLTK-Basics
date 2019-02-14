from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

ps=PorterStemmer()

ex_words=["python","pythoner","pythoning","pythoned","pythonly"]

for w in ex_words:
    print(ps.stem(w))

text="It is very important to be pythonly while you are pythoning with python"

words=word_tokenize(text)

for w in words:
    print(ps.stem(w))
