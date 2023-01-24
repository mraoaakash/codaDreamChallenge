import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf


basepath = "/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/Train/meta_data"

patient_data = pd.read_csv(os.path.join(basepath, "CODA_TB_Clinical_Meta_Info.csv"))
audio_data = pd.read_csv(os.path.join(basepath, "CODA_TB_Solicited_Meta_Info.csv"))
patient_data = patient_data[['participant','tb_status']]
new_df = pd.merge(patient_data, audio_data, on='participant')
new_df = new_df[['participant','tb_status','filename','sound_prediction_score']]
  

# reading spectrograms to csv
specdf = pd.DataFrame(columns=['spectrogram'])
specpath = "/home/chs.rintu/Documents/chs-lab-ws02/research-challenges/dream/coda-tb-22/Train/raw_data/spect"
for file in os.listdir(specpath):
    if file.endswith(".png"):
        specdf = specdf.append({'spectrogram': file}, ignore_index=True)
specdf.to_csv(os.path.join(basepath, "spectrograms.csv"), index=False)

# Sorting the datasets
new_df = new_df.sort_values(by=['filename'])
new_df = new_df.reset_index(drop=True)
specdf = specdf.sort_values(by=['spectrogram'])
specdf = specdf.reset_index(drop=True)
new_df['spectrogram'] = specdf['spectrogram']


# Individual datasets for positive and negative patients
new_df_positive = new_df[new_df['tb_status'] == 1]
new_df_negative = new_df[new_df['tb_status'] == 0]

# Shuffled datasets
new_df_positive = new_df_positive.sample(frac=1).reset_index(drop=True)
new_df_negative = new_df_negative.sample(frac=1).reset_index(drop=True)

# Splitting the datasets into train and test
train_positive = new_df_positive[:int(0.8*len(new_df_positive))]
test_positive = new_df_positive[int(0.8*len(new_df_positive)):]
train_negative = new_df_negative[:int(0.8*len(new_df_negative))]
test_negative = new_df_negative[int(0.8*len(new_df_negative)):]
train_df = pd.concat([train_positive, train_negative])
test_df = pd.concat([test_positive, test_negative])


print(patient_data.head())
print(audio_data.head())
print(new_df.head())
# new_df.to_csv(os.path.join(basepath, "masterData.csv"), index=False)
print(new_df_positive.head())
print(new_df_negative.head())
print(train_positive.head())
print(test_positive.head())
print(train_negative.head())
print(test_negative.head())
print(train_df.head())
print(test_df.head())


# Saving the test and train datasets
train_df.to_csv(os.path.join(basepath, "trainData.csv"), index=False)
test_df.to_csv(os.path.join(basepath, "testData.csv"), index=False)
