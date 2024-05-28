import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
intends = json.load(open('intends.json'))
# stop_words = set(stopwords.words("english"))
all_words = []
filtered_words = []
tags = []
documents = []
ignore_letters = [',','.','!',"$",'?']

def stem(word):
    return lemmatizer.lemmatize(word.lower())

def tokernize(token_word):
    return nltk.word_tokenize(token_word)

for intent in intends['intents']:
    tag = intent['tag']
    tags.append(tag)

    for pattern in intent['patterns']:
        wordlist = tokernize(pattern)
        all_words.extend(wordlist)
        documents.append((wordlist, tag))

all_words = [stem(word) for word in all_words if word not in ignore_letters]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

pickle.dump(all_words, open('words.pkl', 'wb'))
pickle.dump(tags, open('tags.pkl', 'wb'))

training = []
outputEmpty = [0] * len(tags)

for  (document, tag)  in documents:
    bag = []
    wordPatterns = document

    wordPatterns = [stem(word) for word in wordPatterns]
    for word in all_words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    outputRow = list(outputEmpty)

    outputRow[tags.index(tag)] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(all_words)]
trainY = training[:, len(all_words):]

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))

sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print('Done')

