import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

trainingData = pd.read_csv("F:/DisasterTweets/data/train.csv")


tweetText = trainingData['text'].values

#X-data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tweetText)
tokenized_words = tokenizer.texts_to_sequences(tweetText)

max_length = max(len(seq) for seq in tqdm(tokenized_words, desc="Nuh uh: "))
padded_words = pad_sequences(tokenized_words, maxlen=max_length)

#Y-data - already in 1, 0 format
tweet_type = trainingData['target'].values

x_train, x_test, y_train, y_test = train_test_split(padded_words, tweet_type, random_state=42, test_size=0.2)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
