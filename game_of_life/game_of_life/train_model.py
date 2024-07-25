# import libraries
import numpy as np
import os
import keras
import tf2onnx
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import tensorflow as tf

# set path
os.chdir(r"/Users/cr/Documents/UU/Master/thesis/life")

"""
keras version: 2.15.0
tf2onnx version: 1.16.1
tensorflow version: 2.15.0
"""

def train_model(data_size = int, plot_time_step = 10):
    # Set the seed for Python's built-in random module
    random.seed(66)

    # Set the seed for NumPy
    np.random.seed(66)

    # Set the seed for TensorFlow
    tf.random.set_seed(66)
    
    # load data
    X_train_d = np.load('X_train_d.npy')
    y_train_d = np.load('y_train_d.npy')
    X_val_d = np.load('X_val_d.npy')
    y_val_d = np.load('y_val_d.npy')
    
    X_train_s = np.load('X_train_s.npy')
    y_train_s = np.load('y_train_s.npy')
    X_val_s = np.load('X_val_s.npy')
    y_val_s = np.load('y_val_s.npy')
    
    X_test = np.load('X_test.npy')

    # trim data
    r = int(((data_size*0.8)/2))
    t = int(((data_size*0.2)/2))
    
    X_train_d = X_train_d[0:r, ]
    y_train_d = y_train_d[0:r, ]
    X_val_d = X_val_d[0:t, ]
    y_val_d = y_val_d[0:t, ]

    X_train_s = X_train_s[0:r, ]
    y_train_s = y_train_s[0:r, ]
    X_val_s = X_val_s[0:t, ]
    y_val_s = y_val_s[0:t, ]


    # Ensure the data is in the correct format
    X_train_d = X_train_d.astype(np.float32)
    y_train_d = y_train_d.astype(np.float32)
    X_val_d = X_val_d.astype(np.float32)
    y_val_d = y_val_d.astype(np.float32)

    X_train_s = X_train_s.astype(np.float32)
    y_train_s = y_train_s.astype(np.float32)
    X_val_s = X_val_s.astype(np.float32)
    y_val_s = y_val_s.astype(np.float32)

    X_test = X_test.astype(np.float32)


    # create model

    # CNN Properties
    filters = 50
    kernel_size = (3, 3) # look at all 8 neighboring cells, plus itself
    strides = 1
    hidden_dims = 100

    model = Sequential()
    model.add(Conv2D(
        filters,
        kernel_size,
        padding = 'same',
        activation = 'relu',
        strides=strides,
        input_shape=(20, 20, 1)
    ))
    model.add(Dense(hidden_dims))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss = 'binary_crossentropy', 
                  optimizer = 'adam', 
                  metrics = ['accuracy'])

    # Fit the model
    # dynamic fit
    model.fit(X_train_d, y_train_d,
              validation_data=(X_val_d, y_val_d),
              epochs = 50,
              batch_size = 32)
    
    # static fit
    model.fit(X_train_s, y_train_s,
            validation_data = (X_val_s, y_val_s),
            epochs = 50,
            batch_size = 32)
    
    # predict
    y_pred = X_test[0].reshape(1, 20, 20, 1)
    Y_pred = np.zeros(shape = (100, 20, 20, 1))

    for i in range(100): 
        Y_pred[i] = y_pred
        y_pred = (model.predict(y_pred) >= 0.5).astype(float)

    for i in range(plot_time_step+1): 
        fig, axes = plt.subplots(1, 2, figsize=(10, 8))  # Create 2 subplots in a single column

        # Plot the first raster plot
        axes[0].imshow(X_test.reshape((100, 20, 20))[i])
        axes[0].set_title(f'PCRaster - Time Step {i}')

        # Plot the second raster plot
        axes[1].imshow(Y_pred.reshape((100, 20, 20))[i])
        axes[1].set_title(f'ML Model {data_size} - Time Step {i}')

        plt.tight_layout()
        plt.show()

    # remove time step 0 data, which is the same as the input
    X_test = X_test[1:]
    Y_pred = Y_pred[1:]

    # reshape data foe confusion matrix
    y_true = X_test.reshape(-1)
    y_pred = Y_pred.reshape(-1)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # Print the results
    print(f"Confusion Matrix, data size {data_size}:")
    print(f"Accuracy: {np.mean(accuracy):.2f}")
    print(f"Recall: {np.mean(recall):.2f}")
    print(f"Specificity: {np.mean(specificity):.2f}")

    # Save the Keras model
    model.save('cnn_game_of_life.h5')

    keras.saving.load_model('cnn_game_of_life.h5', custom_objects=None, compile=True, safe_mode=True)

    # Convert to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(model)

    # Save the ONNX model
    # Determine the filename based on data size
    filename = f"cnn_game_of_life{data_size}.onnx" 

    # Save the ONNX model to the determined filename
    with open(filename, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print("ONNX model saved as", filename)

train_model(data_size = 200, # set the data size to train the model
            plot_time_step = 10) # set how many time steps to plot