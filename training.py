import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
# Work, Working, Worked

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.optimizers import SGD   # Stochastic Gradient Descent

lemmatizer = WordNetLemmatizer() # First of all, all words and phrases need to be condensed using the lemmatizer
intents = json.load(open().read) # This is the most complicated part we need to figure out because we are trying to access the customer data


words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ','] # We need to ignore those punctuation marks that are quite far more sensitive.


# A little bit about the json file


# We have to iterate on a json file to process each request
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#          word_list = nltk.word_tokenize(pattern);
#          words.extend(word_list);
#          documents.append((word_list, intent['tag']))
#          if intent['tag'] not in classes:
#             classes.append(intent['tag])
#
# We want to select these words without the punctuation marks
# words = [lemmatizer.lemmatize(w) for w in words if w not in ignore_letters]
# words = sorted(set(words)) # The main purpose of set() function is used to eliminate the duplicates,
                           # and the sorted function sorts each word in an alphabetical order.
# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(words, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0] # Focus on the first item in the dictionary
    word_patterns = [lemmatizer.lemmatize(word.lower) for word in word_patterns]
    for word in words:
        if word in word_patterns:
            bag.append(1)
        else:
            bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])


# Neural Network
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0]) # bag
train_y = list(training[:, 1]) # output row

model = Sequential()
model.add(Dense(128, input_shape = (len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64), activation='relu')
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0])), activation='softmax')

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, neterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size= 5, verbose = 1)
model.save('chatbot_model.model', hist)
print("Done")


