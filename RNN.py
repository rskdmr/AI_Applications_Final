# Ryan Skidmore
# ITP 259 Fall 2023
# Final Project Part 2

import tensorflow
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# Problem 1:
data = pd.read_csv("english-spanish-dataset.csv")
eng = data["english"][:50000]
span = data["spanish"][:50000]
# Problem 2:
# Problem 3:
# English Tokenier
eng_t = Tokenizer()
eng_t.fit_on_texts(eng)
eng_sequences = eng_t.texts_to_sequences(eng)

# Spanish Tokenizer
span_t = Tokenizer()
span_t.fit_on_texts(span)
span_sequences = span_t.texts_to_sequences(span)

# Problem 4:
# English Sequence Padding
eng_pad = pad_sequences(eng_sequences, maxlen=12, padding="post")
# Spanish Sequence Padding
span_pad = pad_sequences(span_sequences, maxlen=12, padding="post")

print(eng_pad.shape)
print(span_pad.shape)

# Reshape Pads for GRU layer compatibility:
eng_pad = eng_pad.reshape(*eng_pad.shape, 1)
span_pad = span_pad.reshape(*span_pad.shape, 1)

print(eng_pad)
print(span_pad.shape)

# Problem 5:
print(eng_pad.shape[1], eng_pad.shape[2])
print(eng_pad.shape[1:])
model = Sequential()
# model.add(GRU(64, input_shape=(eng_pad.shape[1], eng_pad.shape[2])))
model.add(GRU(128, input_shape=eng_pad.shape[1:], return_sequences=True))
model.add(Dropout(0.25))
model.add(GRU(64, input_shape=eng_pad.shape[1:], return_sequences=True))
model.add(Dropout(0.3))
model.add(Dense(units=len(span_t.word_index) + 1, activation="softmax"))

model.summary()

# Problem 6:
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
early_stopping = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)

# Problem 7:
h = model.fit(eng_pad, span_pad, batch_size=150, epochs=10, validation_split=0.2, callbacks=early_stopping)

# Problem 8:GG
pd.DataFrame(h.history).plot()
plt.show()

# Problem 9
user_input = input("Please enter a sentence to be translated to Spanish: ")
user_input = [eng_t.word_index[word] for word in user_input.split()]
print(user_input)
# Tokenize the User input Data based on the tokenized English phrases:
# user_sequences = eng_t.texts_to_sequences([user_input])

# Pad the User input Data:
user_pad = pad_sequences([user_input], maxlen=12, padding="post")
# Reshape the input:
new_user_pad = []
for item in user_pad:
    for thing in item:
        new_user_pad.append([thing])

# Predict on the padded input:
print(new_user_pad)
user_pad.reshape(-1, 1)
print(user_pad)
pred = model.predict(user_pad)
print(pred.shape)
print(len(pred[0][0]))


# Problem 10
# Change the predicted softmax output back into Spanish:
def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    """join_list = []
    for item in logits:
        print(item)
        print(type(item))
        predict = np.argmax(item, 1)
        print(predict)
        # if predict == 0:
            # join_list.append("PAD")
        #else:
    predict[0] = '<PAD>'
    join_list.append(index_to_words[predict])
    return ' '.join(join_list)"""
    index_to_words[0] = "<PAD>"
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])


print(logits_to_text(pred[0], span_t))
