from nltk import word_tokenize,sent_tokenize
import string
text="Hello Mr. Smith How are you? What are your plans today? I want to show you something."

#print(sent_tokenize(text))
#print(word_tokenize(text))
k=string.punctuation(text)
for i in (sent_tokenize(text)):
    print(i)
for j in (word_tokenize(text)):
    print(j)
