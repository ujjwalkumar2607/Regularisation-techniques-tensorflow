import tensorflow as tf
print(tf.__version__)
# Load the diabetes dataset
from sklearn.datasets import load_diabetes
diabetes_dataset = load_diabetes()
print(diabetes_dataset['DESCR'])

print(diabetes_dataset.keys())
print(diabetes_dataset['data'][0])
print(diabetes_dataset['target'][0])

data = diabetes_dataset['data']
targets = diabetes_dataset['target']

# Normalise the target data (this will make clearer training curves)
targets = (targets - targets.mean(axis = 0)) / targets.std()
targets

# Split the data into train and test sets
from sklearn.model_selection import train_test_split

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size = 0.1)

print(train_data.shape)
print(test_data.shape)
print(train_targets.shape)
print(test_targets.shape)

# Build the model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def get_model():
    model = Sequential([
        Dense(units = 128, activation = 'relu', input_shape = (train_data.shape[1],)),
        Dense(units = 128, activation = 'relu'),
        Dense(units = 128, activation = 'relu'),
        Dense(units = 128, activation = 'relu'),
        Dense(units = 128, activation = 'relu'),
        Dense(units = 128, activation = 'relu'),
        Dense(units = 1)
    ])
    return model
model = get_model()

model.summary()

# Compile the model
model.compile(optimizer = 'adam',
             loss = 'mse',
             metrics = ['mae', 'accuracy'])

# Train the model, with some of the data reserved for validation
history = model.fit(train_data, train_targets, epochs = 100, validation_split = 0.15, batch_size = 64)

history.history.keys()

# Evaluate the model on the test set

model.evaluate(test_data, test_targets)

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
# %matplotlib inline

# Plot the training and validation loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers

def get_regularised_model(wd, rate):
    model = Sequential([
        Dense(128, kernel_regularizer = regularizers.l2(wd), activation="relu", input_shape=(train_data.shape[1],)),
        Dropout(rate),
        Dense(128, kernel_regularizer = regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer = regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer = regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer = regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer = regularizers.l2(wd), activation="relu"),
        Dropout(rate),
        Dense(1)
    ])
    return model

# Re-build the model with weight decay and dropout layers
model = get_regularised_model(1e-5, 0.3)

# Compile the model

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])

# Train the model, with some of the data reserved for validation
history = model.fit(train_data, train_targets, epochs = 100, validation_split = 0.15, batch_size = 64)

# Evaluate the model on the test set

model.evaluate(test_data, test_targets)

# Plot the training and validation loss

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(patience = 2)

# Re-train the unregularised model
unregularised_model = get_model()
unregularised_model.compile(optimizer = 'adam', loss = 'mse')
unreg_history = unregularised_model.fit(train_data, train_targets, epochs = 100, validation_split = 0.15,
                                        batch_size = 64, callbacks = [early_stopping])

# Evaluate the model on the test set
unregularised_model.evaluate(test_data, test_targets, verbose = 1)

# Re-train the regularised model
regularised_model = get_regularised_model(1e-8, 0.2)
regularised_model.compile(optimizer = 'adam', loss = 'mse')
reg_history = regularised_model.fit(train_data, train_targets, epochs = 100,
                                   validation_split = 0.15, batch_size = 64, 
                                   callbacks = [early_stopping])

# Evaluate the model on the test set
regularised_model.evaluate(test_data, test_targets, verbose = 1)

# Plot the training and validation loss

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 5))

fig.add_subplot(121)

plt.plot(unreg_history.history['loss'])
plt.plot(unreg_history.history['val_loss'])
plt.title('Unregularised model: loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

fig.add_subplot(122)

plt.plot(reg_history.history['loss'])
plt.plot(reg_history.history['val_loss'])
plt.title('Regularised model: loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.show()
