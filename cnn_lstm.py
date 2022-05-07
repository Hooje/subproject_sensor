import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import *
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# change conv1D to lstm 

with open('combined.pickle', 'rb') as f:
    Xy = pickle.load(f)
X = Xy[:,:-1]
y = Xy[:,-1]
#y = np.random.randint(2, size = len(y))
#input(y)
#input(y.shape)
#y = np.randomint
x_train, x_test, y_train, y_test = train_test_split\
(X, y, test_size=0.33, random_state=42)



TIME_PERIODS = 100
num_sensors = 1
input_shape = 100
num_classes = 2


#input(set(y_train))

model_m = Sequential()
model_m.add(Reshape((TIME_PERIODS, num_sensors), input_shape=(input_shape,)))
model_m.add(LSTM(units = 50,return_sequences = True))
model_m.add(Dropout(0.5))
model_m.add(MaxPooling1D(3))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(LSTM(units = 50))
model_m.add(Dense(num_classes, activation='softmax'))

print(model_m.summary())

callbacks_list = [
keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

model_m.compile(loss="sparse_categorical_crossentropy",
optimizer='adam', metrics=['accuracy'])

BATCH_SIZE = 10
EPOCHS = 50

history = model_m.fit(x_train,
y_train,
batch_size=BATCH_SIZE,
epochs=EPOCHS,
callbacks=callbacks_list,
validation_split=0.2,
verbose=1)

y_p = model_m.predict(x_test)
y_p = [np.argmax(i) for i in y_p]
y_t = [int(i) for i in y_test]

acc = accuracy_score(y_p, y_t)
print(y)
print(y_p)

print(acc)