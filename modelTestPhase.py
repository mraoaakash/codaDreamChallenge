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

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from tensorflow.keras.preprocessing.image import ImageDataGenerator
   
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


EPOCHS = 30
BATCH_SIZE = 128
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
INP_SIZE = (IMG_SIZE, IMG_SIZE)


def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

# do lr optimization
# increase model complexity
# fix_gpu()

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

# mobnet = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
# for layer in mobnet.layers:
#     layer.trainable = True
# x = Flatten()(mobnet.output)
# x = Dropout(0.2)(x)
# x = Dense(2, activation = 'softmax')(x)
# model1 = Model(mobnet.input, x)
# model1.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

inception = InceptionV3(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
for layer in inception.layers:
    layer.trainable = True
input_layer = Input(shape=(224,224,3)) #Image resolution is 224x224 pixels
x = Conv2D(128, padding='same', activation='relu', strides=(2, 2))(input_layer)
x = Conv2D(128, padding='same', activation='relu', strides=(2, 2))(x)
x = Conv2D(64, padding='same', activation='relu', strides=(2, 2))(x)
x = MaxPool2D(padding='same',strides=(2, 2))(x)
x = Conv2D(64, padding='same', activation='relu', strides=(2, 2))(x)
x = Conv2D(64, padding='same', activation='relu', strides=(2, 2))(x)
x = MaxPool2D( padding='same', strides=(2, 2))(x)
x = GlobalAveragePooling2D()(x)

predictions = Dense(11, activation='softmax')(x) #I have 11 classes of image to classify

model = Model(inputs = input_layer, outputs=predictions)

model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# effnet = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
# for layer in effnet.layers:
#     layer.trainable = True
# x = Flatten()(effnet.output)
# x = Dropout(0.2)(x)
# x = Dense(2, activation = 'softmax')(x)
# model3 = Model(effnet.input, x)
# model3.compile(optimizer = RMSprop(learning_rate = 0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# Building the training data generator

train_path = ''

datagen_train = ImageDataGenerator(rescale = 1./255,
                                    rotation_range = 40,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True,
                                    fill_mode = 'nearest')

datagen_test = ImageDataGenerator(rescale = 1./255)

train_generator = datagen_train.flow_from_dataframe(
        train_df,
        directory = "/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/Train/raw_data/spect",
        x_col='filename',
        y_col='label',
        target_size=INP_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical')
#Validation Data
valid_generator = datagen_test.flow_from_dataframe(
        test_df,
        directory = "/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/Train/raw_data/spect",
        x_col='filename',
        y_col='label',
        target_size=INP_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False)

for model_type, model in zip(['inception'], [ model2]):
    print("------------------------------------------")
    print(f'Training the model {model_type}')
    print("------------------------------------------")
    filepath = f'/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/models/{model_type}/modellog'
    if os.path.exists(filepath):
        os.makedirs(filepath)
    filepath = filepath + "/model-{epoch:02d}-{val_acc:.2f}.h5"
    callbacks = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    history = model.fit(train_generator, validation_data = valid_generator, epochs=EPOCHS, verbose=1, callbacks=callbacks)

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
    with open(hist_json_file, mode='w+') as f:
        hist_df.to_json(f)

    # or save to csv: 
    hist_csv_file = f'/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/models/{model_type}/history.csv'
    with open(hist_csv_file, mode='w+') as f:
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