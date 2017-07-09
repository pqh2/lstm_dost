from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM
import numpy as np 
import random


enc = {}
dec = {}

i = 0
chars = ' abcdefghijklmnopqrstuwvxyzABCDEFGHJKLMNOPQRSTUVXWYZ.!?@#$%^&*()'
for c in chars:
    enc[c] = i
    dec[i] = c
    i += 1


training_text = ""
with open("dost.txt") as f:
  while True:
    c = f.read(1)
    if not c:
      print "End of file"
      break
    if c in enc:
        training_text += c
    else:
        training_text += " "

x_train = []
y_train = []

print len(training_text)
for i in range(len(training_text) - 61):
    seq = []
    for j in range(60):
        seq.append(training_text[i + j])
    #print seq
    encoded = [enc[c] for c in seq]
    #print encoded  
    x_train.append(np.array(encoded))
    out = [0 for j in range(len(enc))]
    #print out
    #print len(out)
    #print training_text[i + 10]
    #print enc[training_text[i + 10]] 
    out[enc[training_text[i + 60]] if training_text[i + 60] in enc else 0] = 1
    y_train.append(np.array(out))
    if i % 100 == 0:
        print i

x_train = np.array(x_train)
y_train = np.array(y_train)
        



model = Sequential()
model.add(Embedding(len(enc), 40, input_length=60))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
#model.add(LSTM(128))
model.add(Dropout(0.25))
model.add(Dense(len(enc), activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(0.001),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=256, nb_epoch=5, verbose=1, shuffle=True)
#score = model.evaluate(x_test, y_test, batch_size=16)


generated_text = "He had not far to go; he knew indeed how many steps it was from the gate of his lodging house: exactly seven hundred and thirty. He had counted"
generated_text = generated_text[:140]


for i in range(1000):
    sliced = generated_text[-60:]
    encoded = np.reshape(np.array([enc[c] for c in sliced]), (1,60))
    pred = model.predict(encoded)
    above_05 = []
    #print len(pred[0])
    for j in range(len(pred[0])):
        if pred[0][j] > 0.1:
            above_05.append(j)
    print len(above_05)
    random.shuffle(above_05)
    v = above_05[0] if above_05 else np.argmax(pred[0]) 
    generated_text += dec[v]
    print generated_text

