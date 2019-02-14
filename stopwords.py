from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

sent="This is an example showing off word filtration"

stop_words=set(stopwords.words("english"))

words=word_tokenize(sent)

l=[]

for w in words:
    if w not in stop_words:
        l.append(w)

print(l)
print(stop_words)
