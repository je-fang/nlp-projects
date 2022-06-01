#!/usr/bin/env python
# coding: utf-8

# In[50]:


import spacy
from spacy import displacy


# In[4]:


get_ipython().system('python -m spacy download en_core_web_sm')


# In[51]:


nlp = spacy.load('en_core_web_sm')


# In[32]:


txt = "Hello!

I am very interested in the wide world of languages beyond my native English, even if I don't speak any others very well. Recently, I have become rather interested in Romanian, which has managed to hold my attention longer than the other languages I made short endeavours into learning the basics of, which is now what I intend to do with Romanian.

There is no big reason I am interested in learning Romanian, I am not myself of Romanian descent, I know no Romanians, and I have minimal interest in visiting Romania one day, but the language has such an interesting sound, look, and style to it that I would like to puzzle out the basics, at least, and DuoLingo leaves something to be desired.

What I am not looking for is a phrasebook, dictionary, or "Learn Romanian in (x) Days!" type of book. I would like a book that breaks down the grammar, writing, and pronunciation of Romanian, something that clearly presents the basics of the language, even if it's only the barebones minimum. Something that provides a foundation to build off of.

Of course, there are plenty of other ways to learn/practice Romanian (Conversations with native speakers, taking in Romanian Language media, etc.), as with any other language, but I have always been a "book learner", and this whole subreddit is for books, so I'd figured I'd cast my nets here, in addition to other resources.

Thank you in advance."
doc = nlp(txt)


# In[33]:


displacy.render(doc, style='ent')


# In[17]:


import requests
import pandas as pd

with open('reddit_ids.txt', 'r') as fp:
    lines = fp.read().splitlines()
    
    login = {'grant_type': 'password',
                'username': lines[0],
                'password': lines[1]}

    auth = requests.auth.HTTPBasicAuth(lines[2], lines[3])


# In[18]:


headers = {'User-Agent' : 'NERtest/0.0.1'}

res = requests.post(f'https://www.reddit.com/api/v1/access_token',
                    auth=auth, data=login, headers=headers)

res

token = res.json()['access_token']

headers['Authorization'] = f'bearer {token}'


# In[19]:


requests.get('https://oauth.reddit.com/api/v1/me', headers=headers)


# In[43]:


api = 'https://oauth.reddit.com'

res = requests.get(f'{api}/r/investing/new', headers=headers, params={'limit': '100'})


# In[44]:


df = pd.DataFrame({
    'id': [],
    'created_utc': [],
    'subreddit': [],
    'title': [],
    'selftext': [],
    'upvote_ratio': [],
    'ups': [],
    'downs': [],
    'score': []
})

for post in res.json()['data']['children']:
    
    df = df.append({
        'id': post['data']['name'],
        'created_utc': post['data']['created_utc'],
        'subreddit': post['data']['subreddit'],
        'title': post['data']['title'],
        'selftext': post['data']['selftext'],
        'upvote_ratio': post['data']['upvote_ratio'],
        'ups': post['data']['ups'],
        'downs': post['data']['downs'],
        'score': post['data']['score']
    }, ignore_index = True)


# In[45]:


while True:
    
    res = requests.get(f'{api}/r/investing/new', headers=headers, 
                       params={'limit': '100', 'after': df['id'].iloc[len(df)-1]})

    if len(res.json()['data']['children']) == 0:
         break
    
    for post in res.json()['data']['children']:

        df = df.append({
            'id': post['data']['name'],
            'created_utc': post['data']['created_utc'],
            'subreddit': post['data']['subreddit'],
            'title': post['data']['title'],
            'selftext': post['data']['selftext'],
            'upvote_ratio': post['data']['upvote_ratio'],
            'ups': post['data']['ups'],
            'downs': post['data']['downs'],
            'score': post['data']['score']
        }, ignore_index = True)


# In[46]:


df


# In[42]:


#df = df.replace({'|': ''}, regex=True)

#df.to_csv('reddit_posts.csv', sep='|', index=False)


# In[56]:


BLACKLIST = ['ev', 'covid', 'etf', 'nyse', 'sec', 'spac', 'fda', 'fidelity', 'fed', 'treasury', 'etfs']

def get_orgs(text):
    
    doc = nlp(text)
    org_list = []
    
    for entity in doc.ents:
        if entity.label_ == 'ORG' and entity.text.lower() not in BLACKLIST:
            org_list.append(entity.text)
            
    org_list = list(set(org_list))
    return org_list


# In[57]:


df['organizations'] = df['selftext'].apply(get_orgs)
df.head()


# In[58]:


# merge organizations column into one big list
orgs = df['organizations'].to_list()
orgs = [org for sublist in orgs for org in sublist]
orgs[:10]


# In[59]:


from collections import Counter

org_freq = Counter(orgs)
org_freq.most_common(10)


# In[60]:


df.to_csv('reddit_ner.csv', sep='|', index=False)


# In[ ]:




