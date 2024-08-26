

#%%
# Importing the libraries
import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import keras
import cv2
import numpy as np
from keras import layers
import keras_tuner as kt

import itertools
from tensorflow.keras.callbacks import History
from sklearn.metrics import confusion_matrix
import seaborn as sns


# %%
file_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(file_dir)
# %%
# Function that loads the image from the file path and normalizes it;
def load_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255

    return image

#%%
# Loads the images in the 'Image' column of the csv_file dataframe and maps them to the labels in the 'Class' column
def load_dataset(csv_file, image_dir):
    df = pd.read_csv(csv_file)
    images = []
    labels = []
    for index, row in df.iterrows():
        file_path = os.path.join(image_dir, row['Image'])
        image = load_image(file_path)
        images.append(image)
        label = row['Class']
        labels.append(label)
    
    labels = np.array(labels)
    images = np.array(images)
    
    indices = np.arange(images.shape[0])
    np.random.shuffle(indices)
    
    images = images[indices]
    labels = labels[indices]
    
    labels = tf.keras.utils.to_categorical(labels, 100)
    return images, labels

#%%
train_dataset = load_dataset('train.csv', 'train_images')
val_dataset = load_dataset('val.csv', 'val_images')
#%%
# Simple network; Used as a starting point
def simple_mlp():
    model = Sequential(
        [
            Flatten(input_shape=(64, 64, 3)),
            Dense(64, activation='relu'),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(100, activation='softmax')
        ]
    )

    model.compile(optimizer = SGD(learning_rate=0.005),
                  loss='categorical_crossentropy',
                  metrics=[keras.metrics.F1Score()])
    
    return model

# %%
mlp = simple_mlp()
# %%
train_images, train_labels = train_dataset
val_images, val_labels = val_dataset

# %%

history = mlp.fit(
    train_images, train_labels,
    epochs=20,
    validation_data=(val_images, val_labels),
    verbose=1
)
#Very poor performance; 0% precision and recall
# %%
# Save the results
hist_df = pd.DataFrame(history.history)

# Save to CSV
hist_df.to_csv('training_history.csv', index=False)
# %%
# MLP function for hyperparameter tuning
def build_model(hp_lr, hp_neurons, hp_optimizer):
    model = Sequential([
        Flatten(input_shape=(64, 64, 3)),
        Dense(hp_neurons, activation='relu'),
        Dense(hp_neurons, activation='relu'),
        Dense(100, activation='softmax')
    ])
    optimizer = None
    if hp_optimizer == 'adam':
        optimizer = Adam(learning_rate=hp_lr)
    elif hp_optimizer == 'sgd':
        optimizer = SGD(learning_rate=hp_lr)
    elif hp_optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=hp_lr)
    else:
        raise ValueError("Unknown optimizer")

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=[keras.metrics.F1Score()])
    return model

# I chose to tune the learning rate, the number of neurons and the optimizer
param_grid = {
    'hp_lr': [0.01, 0.005, 0.001],
    'hp_neurons': [64, 128, 256],
    'hp_optimizer': ['adam', 'sgd', 'rmsprop']
}
# %%
# saving the results for each combination of hyperparameters
results = []
for params in itertools.product(*param_grid.values()):
    param_dict = dict(zip(param_grid.keys(), params))
    model = build_model(**param_dict)
    
    history = model.fit(train_images[:100], train_labels[:100], validation_data=(val_images[:100], val_labels[:100]), epochs=10, verbose=0)
    results.append((param_dict, history))
# %%
# results[0][1].history['f1_score'][0].shape
num_rows = len(param_grid['hp_lr'])
num_cols = len(param_grid['hp_neurons'])

# Create a figure for each optimizer
for optimizer_name in param_grid['hp_optimizer']:
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.suptitle(f"Performance with Optimizer: {optimizer_name}")
    
    for i, result in enumerate(results):
        lr, neurons, optimizer = result[0]['hp_lr'], result[0]['hp_neurons'], result[0]['hp_optimizer']
        if optimizer != optimizer_name:
            continue
        row = param_grid['hp_lr'].index(lr)
        col = param_grid['hp_neurons'].index(neurons)
        
        ax = axs[row, col]
        ax.plot(result[1].history['loss'], label = 'Train loss')
        ax.plot(result[1].history['val_loss'], label = 'Val loss')
        ax.set_title(f'LR: {lr}, Neurons: {neurons}, Optimizer: {optimizer}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('loss')
        ax.legend()
    plt.show()
#%%
# block
def resnet_block(input_tensor, num_filters, kernel_size=3, strides=1):
    x = Conv2D(num_filters, (kernel_size, kernel_size), strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_filters, (kernel_size, kernel_size), padding='same')(x)
    x = BatchNormalization()(x)

    if strides != 1 or input_tensor.shape[-1] != num_filters:
        shortcut = Conv2D(num_filters, 1, strides=(strides,strides), padding='same')(input_tensor)
        x = Add()([x, shortcut])
    else:
        x = Add()([x, input_tensor])

    x = Activation('relu')(x)
    return x

# %%
# Resnet model for hyperparameter tuning
def build_model(hp_lr, hp_neurons, hp_optimizer):
    inputs = Input(shape=(64, 64, 3))

    x = Conv2D(64, 7, strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    for filters in [64, 128, 256]:
        if filters == 64:
            strides = 1
        else:
            strides = 2
        x = resnet_block(x, filters, strides=strides)
        
        for i in range(2):
            x = resnet_block(x, filters)
            x = Dense(hp_neurons, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(100, activation='softmax')(x)
    model = Model(inputs, outputs)
    
    optimizer = None
    if hp_optimizer == 'adam':
        optimizer = Adam(learning_rate=hp_lr)
    elif hp_optimizer == 'sgd':
        optimizer = SGD(learning_rate=hp_lr)
    elif hp_optimizer == 'rmsprop':
        optimizer = RMSprop(learning_rate=hp_lr)
    else:
        raise ValueError("Unknown optimizer")
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=[keras.metrics.F1Score()])
    return model

param_grid = {
    'hp_lr': [0.01, 0.005, 0.001],
    'hp_neurons': [64, 128, 256],
    'hp_optimizer': ['adam', 'sgd', 'rmsprop']
}
# %%

results = []
for params in itertools.product(*param_grid.values()):
    param_dict = dict(zip(param_grid.keys(), params))
    model = build_model(**param_dict)
    
    history = model.fit(train_images[:100], train_labels[:100], validation_data=(val_images[:100], val_labels[:100]), epochs=10, verbose=0)
    results.append((param_dict, history))
# %%
num_rows = len(param_grid['hp_lr'])
num_cols = len(param_grid['hp_neurons'])

# Create a figure for each optimizer
for optimizer_name in param_grid['hp_optimizer']:
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.suptitle(f"Performance with Optimizer: {optimizer_name}")
    
    for i, result in enumerate(results):
        lr, neurons, optimizer = result[0]['hp_lr'], result[0]['hp_neurons'], result[0]['hp_optimizer']
        if optimizer != optimizer_name:
            continue
        row = param_grid['hp_lr'].index(lr)
        col = param_grid['hp_neurons'].index(neurons)
        
        ax = axs[row, col]
        ax.plot(result[1].history['loss'], label = 'Train loss')
        ax.plot(result[1].history['val_loss'], label = 'Val loss')
        ax.set_title(f'LR: {lr}, Neurons: {neurons}, Optimizer: {optimizer}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('loss')
        ax.legend()
    plt.show()
#%%
# %%
def small_resnet():
    inputs = Input(shape=(64, 64, 3))

    x = Conv2D(64, 7, strides=2, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    for filters in [64, 128, 256]:
        if filters == 64:
            strides = 1
        else:
            strides = 2
        x = resnet_block(x, filters, strides=strides)
        
        for i in range(2):
            x = resnet_block(x, filters)
    
    x = Dense(128, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(100, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

resnet_model = small_resnet()
final_model = resnet_model
# final_model.summary()
# resnet_model.summary()
# %%

final_model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss='categorical_crossentropy',
    metrics=[keras.metrics.F1Score()]
)
# %%

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_cb = ModelCheckpoint("resnet_model.h5", save_best_only=True)
early_stopping_cb = EarlyStopping(patience=10, restore_best_weights=True)

history = resnet_model.fit(
    train_images, train_labels,
    epochs=10,
    batch_size=100,
    validation_data=(val_images, val_labels),
    callbacks=[checkpoint_cb, early_stopping_cb],
    verbose=1
)

# %%
# final_model = mlp
val_labels_pred = final_model.predict(val_images)
val_labels_pred = np.argmax(val_labels_pred, axis=1)
# val_labels = np.argmax(val_labels, axis=1)
# val_labels

conf_matrix = confusion_matrix(val_labels, val_labels_pred)

plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, annot=False, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for MLP')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# %%
# predictions = final_model.predict(train_images[:100])
# predicted_classes = np.argmax(predictions, axis=1)
# predicted_test_classes = np.argmax(train_labels, axis=1)
# predicted_classes[0]
# predicted_test_classes[0]
# %%

# %%
# Loads the images in the 'Image' column of the csv_file
def load_test_data(csv_file='test.csv', image_dir='test_images'):
    df = pd.read_csv(csv_file)
    images = []
    labels = []
    for index, row in df.iterrows():
        file_path = os.path.join(image_dir, row['Image'])
        image = load_image(file_path)
        images.append(image)
    
    images = np.array(images)
    
    return images

test_images = load_test_data()

# %%
predictions = final_model.predict(test_images)
predicted_classes = np.argmax(predictions, axis=1)
# %%
test_df = pd.read_csv('test.csv')
test_df['Class'] = predicted_classes
test_df.head()
test_df.to_csv('submission.csv', index=False)
