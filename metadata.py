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
print(patient_data.head())
print(audio_data.head())
print(new_df.head())
new_df.to_csv(os.path.join(basepath, "masterData.csv"), index=False)