import numpy as np
import os
import time
import datetime
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import EfficientNetB0

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import classification_report, confusion_matrix

train_df = pd.read_csv("/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/Train/meta_data/trainData.csv")[['spectrogram','tb_status']]
test_df = pd.read_csv("/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/Train/meta_data/testData.csv")[['spectrogram','tb_status']]

# changing the column names for train and test
train_df.columns = ['filename','label']
test_df.columns = ['filename','label']

# changing the label names
train_df['label'] = train_df['label'].replace({'positive':1, 'negative':0})
test_df['label'] = test_df['label'].replace({'positive':1, 'negative':0})

# making label as categorical
train_df['label'] = train_df['label'].astype('str')
test_df['label'] = test_df['label'].astype('str')



# shuffling the dataset
train_df = train_df.sample(frac=1).reset_index(drop=True)
test_df = test_df.sample(frac=1).reset_index(drop=True)

# Showing the data head
print(train_df.head())
print(test_df.head())


gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.InteractiveSession(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

mobnet = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(500,500, 3))
for layer in mobnet.layers:
    layer.trainable = True
x = Flatten()(mobnet.output)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(2, activation = 'softmax')(x)
model1 = Model(mobnet.input, x)
model1.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(500,500, 3))
for layer in inception.layers:
    layer.trainable = True
x = Flatten()(inception.output)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(2, activation = 'softmax')(x)
model2 = Model(inception.input, x)
model2.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(500,500, 3))
for layer in effnet.layers:
    layer.trainable = True
x = Flatten()(effnet.output)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(2, activation = 'softmax')(x)
model3 = Model(effnet.input, x)
model3.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# Building the training data generator

train_path = ''

datagen_train = ImageDataGenerator(rescale = 1./255,
                                    rotation_range = 40,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True,
                                    fill_mode = 'nearest')

train_generator = datagen_train.flow_from_dataframe(
        train_df,
        directory = "/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/Train/raw_data/spect",
        x_col='filename',
        y_col='label',
        target_size=(500, 500),
        batch_size=5,
        class_mode='categorical',
        subset = 'training')
#Validation Data
valid_generator = datagen_train.flow_from_dataframe(
        test_df,
        directory = "/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/Train/raw_data/spect",
        x_col='filename',
        y_col='label',
        target_size=(500, 500),
        batch_size=5,
        class_mode='categorical',
        subset = 'validation',
        shuffle=False)

for model_type, model in zip(['mobnet', 'inception', 'effnet'], [model1, model2, model3]):
    print("------------------------------------------")
    print(f'Training the model {model_type}')
    print("------------------------------------------")
    history = model.fit(train_generator, validation_data = valid_generator, epochs=30, verbose=1)

    print("------------------------------------------")
    print(f'Training Complete')
    print("------------------------------------------")
    # Creating a directory to save the model paths 

    # Saving the model
    model.save(f'/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/models/{model_type}/dense121_01.h5')
    print("------------------------------------------")
    print(f'Model saved')
    print("------------------------------------------")


    #plotting the accuracy and loss
    print("------------------------------------------")
    print(f'Plotting and supplimentary data')
    print("------------------------------------------")
    plt.figure(figsize=(10, 10))
    plt.plot(history.history['acc'], label='Training Accuracy')
    plt.plot(history.history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.tight_layout()
    plt.savefig(f'/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/models/{model_type}/Accuracy.jpg')

    # np.save('/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/models/{model_type}/history1.npy',history.history)

    hist_df = pd.DataFrame(history.history) 

    # save to json:  
    hist_json_file = f'/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/models/{model_type}/history.json' 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # or save to csv: 
    hist_csv_file = f'/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/models/{model_type}/history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    loaded_model = load_model(f'/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/models/{model_type}/dense121_01.h5')
    outcomes = loaded_model.predict(valid_generator)
    y_pred = np.argmax(outcomes, axis=1)
    # confusion matrix
    confusion = confusion_matrix(valid_generator.classes, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/models/{model_type}/Confusion_matrix.jpg')

    conf_df = pd.DataFrame(confusion, index = ['negative','positive'], columns = ['negative','positive'])
    conf_df.to_csv(f'/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/models/{model_type}/Confusion_matrix.csv')

    # classification report
    target_names = ['negative','positive']
    report = classification_report(valid_generator.classes, y_pred, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f'/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/models/{model_type}/Classification_report.csv')

    print("------------------------------------------")
    print(f'Supplimentary Data Saved')
    print("------------------------------------------")