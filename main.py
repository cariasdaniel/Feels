import csv
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

with open('labeledTrainData.tsv', 'rb') as csvfile:
    train = csv.reader(csvfile, delimiter='\t', quotechar='\"')

    rows = list(train)
    stp = set(stopwords.words("english"))
    i = 0

    for row in rows:
        row[2] = re.sub("[^a-zA-Z]", " ", row[2]).lower().split()
        row[2] = [w for w in row[2] if not w in stp]

    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None,  max_features=5000)