from nltk.stem import WordNetLemmatizer

lemmatizer=WordNetLemmatizer()

print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("pythoning"))
print(lemmatizer.lemmatize("better",'a'))
print(lemmatizer.lemmatize("best",'a'))
print(lemmatizer.lemmatize("running",'v'))
