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




mobnet = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in mobnet.layers:
    layer.trainable = True
x = Flatten()(mobnet.output)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(2, activation = 'softmax')(x)
model1 = Model(mobnet.input, x)
model1.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

inception = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in inception.layers:
    layer.trainable = True
x = Flatten()(inception.output)
x = Dense(1024, activation = 'relu')(x)
x = Dropout(0.2)(x)
x = Dense(2, activation = 'softmax')(x)
model2 = Model(inception.input, x)
model2.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
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
                                    fill_mode = 'nearest',
                                    validation_split = 0.2)

train_generator = datagen_train.flow_from_directory(
        train_path,
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        subset = 'training')
#Validation Data
valid_generator = datagen_train.flow_from_directory(
        train_path,
        target_size=(300, 300),
        batch_size=32,
        class_mode='categorical',
        subset = 'validation',
        shuffle=False)

for model_type, model in zip(['mobnet', 'inception', 'effnet'], [model1, model2, model3]):
    print("------------------------------------------")
    print(f'Training the model {model_type}')
    print("------------------------------------------")
    history = model.fit(train_generator, validation_data = valid_generator, epochs=50)

    print("------------------------------------------")
    print(f'Training Complete')
    print("------------------------------------------")
    # Creating a directory to save the model paths 

    # Saving the model
    model.save(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/dense121_01.h5')
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
    plt.savefig(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/Accuracy.jpg')

    # np.save('/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/history1.npy',history.history)

    hist_df = pd.DataFrame(history.history) 

    # save to json:  
    hist_json_file = f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/history.json' 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f)

    # or save to csv: 
    hist_csv_file = f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)

    loaded_model = load_model(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/dense121_01.h5')
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
    plt.savefig(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/Confusion_matrix.jpg')

    conf_df = pd.DataFrame(confusion, index = ['wdoscc','mdoscc','pdoscc'], columns = ['wdoscc','mdoscc','pdoscc'])
    conf_df.to_csv(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/Confusion_matrix.csv')

    # classification report
    target_names = ['wdoscc','mdoscc','pdoscc']
    report = classification_report(valid_generator.classes, y_pred, target_names=target_names, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv(f'/storage/bic/data/oscc/data/Histology-image-analysis/models/{model_type}/Classification_report.csv')

    print("------------------------------------------")
    print(f'Supplimentary Data Saved')
    print("------------------------------------------")