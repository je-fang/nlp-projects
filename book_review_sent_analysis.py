from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import os

import pandas as pd
import numpy as np

from transformers import BertTokenizer
from transformers import TFAutoModel

api = KaggleApi()
api.authenticate()

api.dataset_download_file('meetnagadia/amazon-kindle-book-review-for-sentiment-analysis',
                          file_name='all_kindle_review .csv',
                          path='./')

with zipfile.ZipFile('all_kindle_review%20.csv.zip', 'r') as zip_ref:
        zip_ref.extractall('./')

os.remove('all_kindle_review%20.csv.zip')

df = pd.read_csv('all_kindle_review .csv')
df.head()

df['rating'].value_counts().plot(kind='bar')

seq_len = 512
num_samples = len(df)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

tokens = tokenizer(df['reviewText'].tolist(), max_length=seq_len, truncation=True,
                   padding='max_length', add_special_tokens=True,
                   return_tensors='np')

tokens['input_ids'][:10]

with open('review-xids.npy', 'wb') as f:
    np.save(f, tokens['input_ids'])
with open('review-xmask.npy', 'wb') as f:
    np.save(f, tokens['attention_mask'])
    
del tokens

arr = df['rating'].values

labels = np.zeros((num_samples, 5))
labels[np.arange(num_samples), arr-1] = 1


with open('review-labels.npy', 'wb') as f:
    np.save(f, labels)

# prep data

with open('review-xids.npy', 'rb') as f:
    Xids = np.load(f, allow_pickle=True)
with open('review-xmask.npy', 'rb') as f:
    Xmask = np.load(f, allow_pickle=True)
with open('review-labels.npy', 'rb') as f:
    labels = np.load(f, allow_pickle=True)

dataset = tf.data.Dataset.from_tensor_slices((Xids, Xmask, labels))

def map_func(input_ids, masks, labels):
    return {'input_ids': input_ids, 'attention_mask': masks}, labels

dataset = dataset.map(map_func)

batch_size = 16
dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)

split = 0.9
size = int((Xids.shape[0] / batch_size) * split)

train_ds = dataset.take(size)
val_ds = dataset.skip(size)

del dataset

tf.data.experimental.save(train_ds, 'train')
tf.data.experimental.save(val_ds, 'val')

if val_ds.element_spec == train_ds.element_spec:
    
    ds = tf.data.experimental.load('train', element_spec=train_ds.element_spec)

val_ds.element_spec

# train model

bert = TFAutoModel.from_pretrained('bert-base-cased')
bert.summary()

input_ids = tf.keras.layers.Input(shape=(512,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(512,), name='attention_mask', dtype='int32')
embeddings = bert.bert(input_ids, attention_mask=mask)[1]
x = tf.keras.layers.Dense(1024, activation='relu')(embeddings)
y = tf.keras.layers.Dense(5, activation='softmax', name='outputs')(x)


model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
model.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

element_spec = ({'input_ids': tf.TensorSpec(shape=(16, 512), dtype=tf.int32, name=None),
                   'attention_mask': tf.TensorSpec(shape=(16, 512), dtype=tf.int32, name=None)},
                    tf.TensorSpec(shape=(16, 5), dtype=tf.float64, name=None))

train_ds = tf.data.experimental.load('train', element_spec=element_spec)
val_ds = tf.data.experimental.load('val', element_spec=element_spec)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3
)

model.save('sentiment_model')

# predict

model = tf.keras.models.load_model('sentiment_model')
model.summary()

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def prep_data(text):
    tokens = tokenizer.encode_plus(text, max_length=512,
                                   truncation=True, padding='max_length',
                                   add_special_tokens=True, return_token_type_ids=False,
                                   return_tensors='tf')
    # tokenizer returns int32 tensors, we need to return float64, so we use tf.cast
    return {'input_ids': tf.cast(tokens['input_ids'], tf.float64),
            'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}

pd.set_option('display.max_colwidth', None)

df = pd.read_csv('test.tsv', sep='\t')
df.head()

df = df.drop_duplicates(subset=['SentenceId'], keep='first')
#df.head()

df['Sentiment'] = None

for i, row in df.iterrows():
    tokens = prep_data(row['Phrase'])
    probs = model.predict(tokens)
    pred = np.argmax(probs)
    df.at[i, 'Sentiment'] = pred

df.head()