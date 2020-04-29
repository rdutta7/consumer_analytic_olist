import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline

order_review_data = pd.read_csv("../input/olist_order_reviews_dataset.csv")
order_review_data.head()
order_review_data.info()
# dropping NAs
order_review_data = order_review_data.dropna(subset=['review_comment_message'])
# Checking to find max number of words
order_review_data['word_count'] = order_review_data.review_comment_message.apply(lambda x: len(str(x).split()))
order_review_data.word_count.max()

# plot
g = sns.FacetGrid(data=order_review_data, col='review_score',height=5, aspect=0.8)
before_remove = g.map(plt.hist, 'word_count', bins=30)
sns.boxplot(x='review_score', y='word_count', data=order_review_data)

# wordcloud
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from nltk.corpus import stopwords
from string import punctuation
from nltk.stem import RSLPStemmer #Stemmer for portugese words.

from nltk.probability import FreqDist
from collections import defaultdict
from heapq import nlargest

stop = stopwords.words('portuguese')
stop.append('nao') #Stopword already have "Não", just adding this because it's appear on dataframe

# lets join all review comments in a single text, separated by review score
text_review_1 = ' '.join(order_review_data[order_review_data["review_score"]==1]["review_comment_message"])
text_review_2 = ' '.join(order_review_data[order_review_data["review_score"]==2]["review_comment_message"])
text_review_3 = ' '.join(order_review_data[order_review_data["review_score"]==3]["review_comment_message"])
text_review_4 = ' '.join(order_review_data[order_review_data["review_score"]==4]["review_comment_message"])
text_review_5 = ' '.join(order_review_data[order_review_data["review_score"]==5]["review_comment_message"])

def visualize(label):
    words = ''
    for msg in order_review_data[order_review_data['review_score'] == label]['review_comment_message']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.figure(figsize=(12,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

visualize(1)
visualize(5)