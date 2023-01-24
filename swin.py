from tensorflow import keras
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=Warning)



try:  # detect TPUs
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()  # TPU detection
    strategy = tf.distribute.TPUStrategy(tpu)
except ValueError:  # detect GPUs
    tpu = False
    strategy = (
        tf.distribute.get_strategy()
    )  # default strategy that works on CPU and single GPU
print("Number of Accelerators: ", strategy.num_replicas_in_sync)

# Model
IMAGE_SIZE = [224, 224] # Change this accordingly. 
MODEL_PATH = "https://tfhub.dev/sayakpaul/swin_large_patch4_window7_224" 

# TPU
BATCH_SIZE = 128  # on Colab/GPU, a higher batch size may throw OOM

# Dataset
CLASSES = [
    "negative",
    "positive",
]  

# Other constants
MEAN = tf.constant([0.485 * 255, 0.456 * 255, 0.406 * 255])  # imagenet mean
STD = tf.constant([0.229 * 255, 0.224 * 255, 0.225 * 255])  # imagenet std
AUTO = tf.data.AUTOTUNE


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
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical')
#Validation Data
valid_generator = datagen_test.flow_from_dataframe(
        test_df,
        directory = "/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/Train/raw_data/spect",
        x_col='filename',
        y_col='label',
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False)

# num_train = tf.data.experimental.cardinality(train_generator)
# num_val = tf.data.experimental.cardinality(valid_generator)

class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )

def get_model(model_url: str, res: int = IMAGE_SIZE[0], num_classes: int = 5) -> tf.keras.Model:
    inputs = tf.keras.Input((res, res, 3))
    hub_module = hub.KerasLayer(model_url, trainable=True)

    x = hub_module(inputs, training=False) 
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inputs, outputs)

EPOCHS = 10
WARMUP_STEPS = 10
INIT_LR = 0.03
WAMRUP_LR = 0.006

TOTAL_STEPS = 600

scheduled_lrs = WarmUpCosine(
    learning_rate_base=INIT_LR,
    total_steps=TOTAL_STEPS,
    warmup_learning_rate=WAMRUP_LR,
    warmup_steps=WARMUP_STEPS,
)

optimizer = keras.optimizers.SGD(scheduled_lrs)
loss = keras.losses.SparseCategoricalCrossentropy()

with strategy.scope(): # this line is all that is needed to run on TPU (or multi-GPU, ...)
    model = get_model(MODEL_PATH)
    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

history = model.fit(train_generator, validation_data=valid_generator, epochs=EPOCHS)

result = pd.DataFrame(history.history)
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
result[["accuracy", "val_accuracy"]].plot(xlabel="epoch", ylabel="score", ax=ax[0])
result[["loss", "val_loss"]].plot(xlabel="epoch", ylabel="score", ax=ax[1])

sample_images, sample_labels = next(iter(valid_generator))

predictions = model.predict(sample_images, batch_size=16).argmax(axis=-1)
evaluations = model.evaluate(sample_images, sample_labels, batch_size=16)

print("[val_loss, val_acc]", evaluations)

plt.figure(figsize=(5 * 3, 3 * 3))
for n in range(15):
    ax = plt.subplot(3, 5, n + 1)
    image = (sample_images[n] * STD + MEAN).numpy()
    image = (image - image.min()) / (
        image.max() - image.min()
    )  # convert to [0, 1] for avoiding matplotlib warning
    plt.imshow(image)
    target = CLASSES[sample_labels[n]]
    pred = CLASSES[predictions[n]]
    plt.title("{} ({})".format(target, pred))
    plt.axis("off")

plt.tight_layout()
plt.show()