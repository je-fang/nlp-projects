#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import flair


# In[ ]:


model = flair.models.TextClassifier.load('en-sentiment')


# In[1]:


def get_sentiment(text):
    sentence = flair.data.Sentence(text)
    model.predict(sentence)
    sentiment = sentence.labels[0]
    return sentiment


# In[3]:


df = pd.read_csv('reddit_ner.csv', sep='|')
df.head()


# In[ ]:


df['sentiment'] = df['selftext'].apply(get_sentiment)
df.head()


# In[ ]:


import ast

df['organizations'] = df['organizations'].apply(lambda x: ast.literal_eval(x))


# In[ ]:


sentiment = {}

for i, row in df.iterrows():
    direction = row['sentiment'].value
    score = row['sentiment'].score
    for org in row['organizations']:
        if org not in sentiment.keys():
            sentiment[org] = {'POSITIVE': [], 'NEGATIVE': []}
        sentiment[org][direction].append(score)


# In[ ]:


avg_sentiment = []

for org in sentiment.keys():
    
    freq = len(sentiment[org]['POSITIVE']) + len(sentiment[org]['NEGATIVE'])
    
    for direction in ['POSITIVE', 'NEGATIVE']:
        score = sentiment[org][direction]
        if len(score) == 0:
            sentiment[org][direction] = 0.0
        else:
            sentiment[org][direction] = sum(score)
    
    total = sentiment[org]['POSITIVE'] - sentiment[org]['NEGATIVE']
    avg = total/freq
    
    avg_sentiment.append({
        'entity': org,
        'positive': sentiment[org]['POSITIVE'],
        'negative': sentiment[org]['NEGATIVE'],
        'frequency': freq,
        'score': avg
    })


# In[ ]:


sentiment_df = sentiment_df[sentiment_df['frequency'] > 3]
sentiment_df


# In[ ]:


sentiment_df.sort_values('score', ascending=False).head(10)

